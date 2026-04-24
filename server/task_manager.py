"""
任务管理器 - 管理推理任务的状态和队列
"""

import uuid
import time
import threading
import queue
from typing import Optional, Dict, List
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceTask:
    """推理任务类"""
    
    def __init__(self, client_sid: Optional[str] = None):
        """
        初始化推理任务
        
        Args:
            client_sid: 客户端SocketIO会话ID
        """
        self.task_id = str(uuid.uuid4())
        self.created_time: float = time.time()
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.client_sid = client_sid
        self.error_message: Optional[str] = None
        self.result_files: List[str] = []  # 生成的音频文件列表
        # 分阶段结果（phase 1: mixed, phase 2: bgm）
        self.phase: int = 0
        self.phase1_files: List[str] = []
        self.phase2_files: List[str] = []
        # 用于在 phase1 完成后等待客户端确认继续
        self._continue_event = threading.Event()
        self._continue_event.set()  # 默认不阻塞；phase1完成时会clear并等待
        self.lock = threading.Lock()
    
    def set_status(self, status: TaskStatus, error_message: Optional[str] = None):
        """设置任务状态"""
        with self.lock:
            self.status = status
            if error_message:
                self.error_message = error_message
            if status == TaskStatus.RUNNING and self.start_time is None:
                self.start_time = time.time()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                self.end_time = time.time()
    
    def set_progress(self, progress: int):
        """设置任务进度（0-100）"""
        with self.lock:
            self.progress = max(0, min(100, progress))
    
    def add_result_file(self, filename: str):
        """添加结果文件"""
        with self.lock:
            if filename not in self.result_files:
                self.result_files.append(filename)

    def set_phase(self, phase: int):
        """设置当前阶段（0/1/2）"""
        with self.lock:
            self.phase = phase

    def set_phase_files(self, phase: int, files: List[str]):
        """设置某阶段的结果文件列表（覆盖写入）"""
        with self.lock:
            if phase == 1:
                self.phase1_files = list(files)
            elif phase == 2:
                self.phase2_files = list(files)

    def wait_for_continue(self, timeout: Optional[float] = None) -> bool:
        """等待客户端确认继续（用于phase1->phase2）"""
        return self._continue_event.wait(timeout=timeout)

    def require_continue(self):
        """进入等待继续状态：清除事件"""
        self._continue_event.clear()

    def continue_next(self):
        """客户端确认继续：设置事件"""
        self._continue_event.set()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        with self.lock:
            now = time.time()
            elapsed_seconds = None
            duration_seconds = None

            if self.start_time is not None:
                if self.end_time is not None:
                    elapsed_seconds = max(0.0, self.end_time - self.start_time)
                    duration_seconds = elapsed_seconds
                else:
                    elapsed_seconds = max(0.0, now - self.start_time)

            return {
                'task_id': self.task_id,
                'status': self.status.value,
                'progress': self.progress,
                'phase': self.phase,
                'created_time': self.created_time,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'elapsed_seconds': elapsed_seconds,
                'duration_seconds': duration_seconds,
                'error_message': self.error_message,
                'result_files': self.result_files.copy(),
                'phase1_files': self.phase1_files.copy(),
                'phase2_files': self.phase2_files.copy(),
            }


class TaskManager:
    """任务管理器类"""
    
    def __init__(self, max_workers: int = 1):
        """
        初始化任务管理器
        
        Args:
            max_workers: 最大并发工作线程数
        """
        self.tasks: Dict[str, InferenceTask] = {}
        self.task_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.running = True
        
        # 启动工作线程
        self._start_workers()
    
    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_thread, daemon=True, name=f"Worker-{i}")
            worker.start()
            self.workers.append(worker)
    
    def _worker_thread(self):
        """工作线程函数"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 退出信号
                    break
                
                # 执行任务（由inference_runner处理）
                # 这里只是占位，实际执行逻辑在inference_runner中
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"工作线程错误: {e}")
    
    def create_task(self, client_sid: Optional[str] = None) -> InferenceTask:
        """
        创建新任务
        
        Args:
            client_sid: 客户端SocketIO会话ID
            
        Returns:
            新创建的任务
        """
        task = InferenceTask(client_sid=client_sid)
        with self.lock:
            self.tasks[task.task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[InferenceTask]:
        """获取任务"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def add_task_to_queue(self, task: InferenceTask):
        """将任务添加到队列"""
        self.task_queue.put(task)
    
    def remove_task(self, task_id: str):
        """移除任务"""
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
    
    def list_tasks(self) -> List[Dict]:
        """列出所有任务"""
        with self.lock:
            return [task.to_dict() for task in self.tasks.values()]
    
    def shutdown(self):
        """关闭任务管理器"""
        self.running = False
        # 发送退出信号给所有工作线程
        for _ in range(self.max_workers):
            self.task_queue.put(None)
        # 等待所有工作线程结束
        for worker in self.workers:
            worker.join(timeout=5)


# 全局任务管理器实例
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """获取全局任务管理器实例"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(max_workers=1)  # 默认单线程执行推理
    return _task_manager
