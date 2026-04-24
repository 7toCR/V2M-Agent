"""
音乐生成服务端 - Flask服务器，接收文件、执行推理、返回音频文件
"""

import os
import time
import threading
import logging
import json
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import atexit

# 支持相对导入和绝对导入
import sys
import os

# 获取当前文件所在目录和项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 确保项目根目录在路径中（优先使用绝对导入）
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保当前目录也在路径中（用于直接导入）
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 尝试多种导入方式
try:
    # 方式1: 绝对导入（优先，当项目根目录在路径中时）
    from server.task_manager import TaskManager, TaskStatus, get_task_manager
    from server.inference_runner import (
        run_inference_task,
        get_audio_file_path,
        AUDIO_OUTPUT_DIR
    )
except ImportError:
    try:
        # 方式2: 相对导入（当作为模块导入时）
        from .task_manager import TaskManager, TaskStatus, get_task_manager
        from .inference_runner import (
            run_inference_task,
            get_audio_file_path,
            AUDIO_OUTPUT_DIR
        )
    except ImportError:
        # 方式3: 直接导入（当在server目录中直接运行时）
        from task_manager import TaskManager, TaskStatus, get_task_manager
        from inference_runner import (
            run_inference_task,
            get_audio_file_path,
            AUDIO_OUTPUT_DIR
        )

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,  # 增加ping超时时间
    ping_interval=25  # 设置ping间隔
)

# 禁用Flask和Werkzeug的HTTP请求日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # 只显示错误级别的日志

# 禁用Flask的默认日志处理器
app.logger.disabled = True
logging.getLogger('werkzeug').disabled = True

# 获取任务管理器
task_manager = get_task_manager()

# 存储上传的lyric内容（临时，等待推理启动）
uploaded_lyric_content: dict = {}
lyric_lock = threading.Lock()


def safe_print(message):
    """安全打印函数"""
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'ignore').decode('ascii')
        print(safe_message)


def _load_status_push_interval(default_interval: float = 2.0) -> float:
    """
    Try to read push interval from client/config_client_server.json.
    Fallback to a safe default if config is missing/invalid.
    """
    config_path = os.path.join(project_root, "client", "config_client_server.json")
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        interval = float(config.get("progress", {}).get("poll_interval", default_interval))
        return max(1.0, interval)
    except Exception:
        return default_interval


TASK_STATUS_PUSH_INTERVAL = _load_status_push_interval()


def build_task_payload(task, extra: dict = None) -> dict:
    payload = task.to_dict()
    if extra:
        payload.update(extra)
    return payload


def notify_client(task_id: str, event: str, data: dict):
    """通知客户端任务状态更新"""
    task = task_manager.get_task(task_id)
    if task and task.client_sid:
        try:
            socketio.emit(event, data, room=task.client_sid)
            # 减少日志输出，只在重要事件时打印
            # safe_print(f"已发送事件 {event} 给客户端 {task.client_sid}")
        except Exception as e:
            safe_print(f"发送通知失败: {e}")


def start_task_status_stream(task_id: str, stop_event: threading.Event):
    """
    Push task status to subscribed client periodically until task completes/fails.
    """
    last_status_key = None
    while not stop_event.wait(TASK_STATUS_PUSH_INTERVAL):
        task = task_manager.get_task(task_id)
        if not task:
            break

        payload = task.to_dict()
        status_key = (
            payload.get("status"),
            payload.get("progress"),
            payload.get("phase"),
            int(float(payload.get("elapsed_seconds") or 0) // 10),
        )
        if status_key != last_status_key:
            notify_client(task_id, "task_status", payload)
            last_status_key = status_key

        if payload.get("status") in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
            break


# SocketIO 事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接事件"""
    # 减少日志输出，只在调试时打印
    # safe_print(f"客户端连接: {request.sid}")
    emit('connected', {'message': '连接成功', 'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接事件"""
    # 减少日志输出，只在调试时打印
    # safe_print(f"客户端断开: {request.sid}")
    pass


@socketio.on('ping')
def handle_ping(data):
    """处理客户端心跳ping"""
    emit('pong', {'timestamp': data.get('timestamp', 0), 'server_time': time.time()})


@socketio.on('subscribe_task')
def handle_subscribe_task(data):
    """客户端订阅任务更新"""
    task_id = (data or {}).get('task_id')
    if not task_id:
        emit('error', {'message': 'task_id不能为空'})
        return
    task = task_manager.get_task(task_id)
    if task:
        task.client_sid = request.sid
        task_payload = task.to_dict()
        emit('subscribed', {
            'task_id': task_id,
            'message': '订阅成功',
            'current_status': task.status.value,
            'task': task_payload,
        })
        emit('task_status', task_payload)
        # 减少日志输出
        # safe_print(f"客户端 {request.sid} 订阅任务 {task_id}")
    else:
        emit('error', {'message': '任务不存在'})


@socketio.on('get_task_status')
def handle_get_task_status(data):
    """客户端请求任务状态"""
    task_id = (data or {}).get('task_id')
    if not task_id:
        emit('error', {'message': 'task_id不能为空'})
        return
    task = task_manager.get_task(task_id)
    if task:
        emit('task_status', task.to_dict())
    else:
        emit('error', {'message': '任务不存在'})


# HTTP API 端点
@app.route('/api/upload_lyric', methods=['POST'])
def upload_lyric():
    """上传lyric文件内容"""
    try:
        # 检查请求内容类型
        if not request.is_json:
            safe_print("❌ 请求不是JSON格式")
            return jsonify({
                'success': False,
                'error': '请求必须是JSON格式'
            }), 400
        
        data = request.get_json()
        if data is None:
            safe_print("❌ 无法解析JSON数据")
            return jsonify({
                'success': False,
                'error': '无法解析JSON数据'
            }), 400
        if not isinstance(data, dict):
            return jsonify({
                'success': False,
                'error': 'JSON根节点必须是对象'
            }), 400
        
        content = data.get('content', '')
        filename = data.get('filename', 'lyric.jsonl')
        output_dir = data.get('output_dir')  # 可选的输出目录（服务器相对路径）

        if not isinstance(content, str):
            return jsonify({
                'success': False,
                'error': '文件内容必须是字符串'
            }), 400

        content = content.strip()
        if not content:
            return jsonify({
                'success': False,
                'error': '文件内容不能为空'
            }), 400
        
        # 创建任务
        task = task_manager.create_task()
        
        # 保存lyric内容和output_dir（临时存储）
        with lyric_lock:
            uploaded_lyric_content[task.task_id] = {
                'content': content,
                'output_dir': output_dir
            }
        
        safe_print(f"✅ 收到文件: {filename}, 任务ID: {task.task_id}")
        safe_print(f"   内容将写入到 sample/lyrics.jsonl（清空原有内容）")
        
        return jsonify({
            'success': True,
            'task_id': task.task_id,
            'message': '文件上传成功'
        })
    except Exception as e:
        safe_print(f"❌ 上传lyric文件失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/start_inference', methods=['POST'])
def start_inference():
    """启动推理任务"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是JSON格式'
            }), 400

        data = request.get_json() or {}
        if not isinstance(data, dict):
            return jsonify({
                'success': False,
                'error': 'JSON根节点必须是对象'
            }), 400
        client_sid = data.get('client_sid')
        task_id = data.get('task_id')  # 可选的task_id
        generate_type = str(data.get('generate_type', 'both')).strip().lower()  # 生成类型：normal/bgm/both

        if generate_type not in {"normal", "bgm", "both"}:
            return jsonify({
                'success': False,
                'error': f"非法 generate_type: {generate_type}"
            }), 400

        # 获取lyric内容和output_dir
        with lyric_lock:
            if task_id:
                # 如果提供了task_id，使用指定的任务
                if task_id not in uploaded_lyric_content:
                    return jsonify({
                        'success': False,
                        'error': f'任务 {task_id} 的lyric文件不存在'
                    }), 404
                lyric_data = uploaded_lyric_content[task_id]
            else:
                # 如果没有提供task_id，使用最后一个上传的内容
                if not uploaded_lyric_content:
                    return jsonify({
                        'success': False,
                        'error': '请先上传lyric文件'
                    }), 400
                task_id = list(uploaded_lyric_content.keys())[-1]
                lyric_data = uploaded_lyric_content[task_id]
        
        # 兼容旧格式（直接是字符串）和新格式（字典）
        if isinstance(lyric_data, dict):
            lyric_content = lyric_data.get('content', '')
            output_dir = lyric_data.get('output_dir')
        else:
            lyric_content = lyric_data
            output_dir = None

        if not isinstance(lyric_content, str) or not lyric_content.strip():
            return jsonify({
                'success': False,
                'error': 'lyric内容为空或格式错误'
            }), 400
        lyric_content = lyric_content.strip()

        # 获取任务
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': '任务不存在'
            }), 404

        if task.status == TaskStatus.RUNNING:
            return jsonify({
                'success': False,
                'error': '任务已在运行中'
            }), 409

        # 更新客户端ID
        if client_sid:
            task.client_sid = client_sid

        task.set_status(TaskStatus.RUNNING)
        task.set_progress(0)

        # 通知客户端任务开始
        notify_client(task_id, 'task_started', build_task_payload(task, {
            'message': '推理任务开始执行'
        }))
        notify_client(task_id, 'task_status', task.to_dict())

        status_push_stop_event = threading.Event()
        status_thread = threading.Thread(
            target=start_task_status_stream,
            args=(task_id, status_push_stop_event),
            daemon=True,
        )
        status_thread.start()

        # 在后台线程中执行推理任务
        def run_task():
            try:
                success = run_inference_task(task, lyric_content, generate_type=generate_type, output_dir=output_dir)

                if success:
                    # 通知客户端任务完成
                    notify_client(task_id, 'task_completed', build_task_payload(task, {
                        'message': '推理任务已完成',
                        'files': task.result_files
                    }))
                else:
                    # 通知客户端任务失败
                    notify_client(task_id, 'task_failed', build_task_payload(task, {
                        'error': task.error_message or '推理任务失败'
                    }))
            except Exception as e:
                task.set_status(TaskStatus.FAILED, str(e))
                notify_client(task_id, 'task_failed', build_task_payload(task, {
                    'error': str(e)
                }))
            finally:
                status_push_stop_event.set()
                notify_client(task_id, 'task_status', task.to_dict())
                # 清理临时存储的lyric内容
                with lyric_lock:
                    if task_id in uploaded_lyric_content:
                        del uploaded_lyric_content[task_id]

        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()

        safe_print(f"🚀 启动推理任务: {task_id}")

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '推理任务已启动'
        })
    except Exception as e:
        safe_print(f"❌ 启动推理任务失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/task/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    
    return jsonify(task.to_dict())


@app.route('/api/task/<task_id>/list_files', methods=['GET'])
def list_task_files(task_id):
    """列出任务生成的所有音频文件"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    phase = request.args.get('phase')
    if phase == '1':
        files = getattr(task, 'phase1_files', task.result_files)
    elif phase == '2':
        files = getattr(task, 'phase2_files', task.result_files)
    else:
        files = task.result_files

    return jsonify({
        'task_id': task_id,
        'phase': getattr(task, 'phase', 0),
        'files': files
    })


@app.route('/api/task/<task_id>/continue', methods=['POST'])
def continue_task(task_id):
    """phase1 下载完成后，客户端调用该接口让服务端继续执行BGM阶段"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'}), 404

    # 只允许在RUNNING且phase==1时继续
    try:
        if getattr(task, 'phase', 0) != 1:
            return jsonify({'success': False, 'error': '当前任务不在可继续阶段'}), 400
        task.continue_next()
        notify_client(task_id, 'task_phase_continued', build_task_payload(task, {
            'message': '已收到继续指令，开始生成BGM阶段'
        }))
        return jsonify({'success': True, 'task_id': task_id, 'message': '已继续执行'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/task/<task_id>/download/<filename>', methods=['GET', 'HEAD'])
def download_audio_file(task_id, filename):
    """下载音频文件，支持 Range 断点续传"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    # 允许在任务运行中下载阶段性产物：
    allowed_files = set(getattr(task, "result_files", []) or [])
    allowed_files.update(getattr(task, "phase1_files", []) or [])
    allowed_files.update(getattr(task, "phase2_files", []) or [])

    if filename not in allowed_files:
        return jsonify({'error': '文件不存在'}), 404

    file_path = get_audio_file_path(filename)
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 404

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)

    # HEAD：仅返回头部
    if request.method == 'HEAD':
        resp = Response(status=200)
        resp.headers['Accept-Ranges'] = 'bytes'
        resp.headers['Content-Length'] = str(file_size)
        resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return resp

    if not range_header:
        # 兼容旧客户端：整文件返回
        try:
            return send_file(file_path, as_attachment=True, download_name=filename)
        except Exception as e:
            safe_print(f"❌ 下载文件失败: {e}")
            return jsonify({'error': str(e)}), 500

    # 解析 Range
    try:
        units, range_spec = range_header.split('=', 1)
        if units.strip().lower() != 'bytes':
            raise ValueError("只支持 bytes Range")

        start_str, end_str = range_spec.split('-', 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)

        if start < 0 or start > end:
            raise ValueError("非法 Range")
    except Exception:
        # Range 解析失败，退回整文件
        try:
            return send_file(file_path, as_attachment=True, download_name=filename)
        except Exception as e:
            safe_print(f"❌ 下载文件失败: {e}")
            return jsonify({'error': str(e)}), 500

    length = end - start + 1

    def generate():
        with open(file_path, 'rb') as f:
            f.seek(start)
            remaining = length
            chunk_size = 1024 * 1024  # 1MB
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    resp = Response(generate(), status=206, mimetype='audio/flac', direct_passthrough=True)
    resp.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
    resp.headers['Accept-Ranges'] = 'bytes'
    resp.headers['Content-Length'] = str(length)
    resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'active_tasks': len(task_manager.tasks)
    })


@app.route('/api/routes', methods=['GET'])
def list_routes():
    """列出所有注册的路由（用于调试）"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify({'routes': routes})


def shutdown():
    """优雅关闭服务器"""
    safe_print("正在关闭服务器...")
    task_manager.shutdown()
    safe_print("服务器已关闭")


if __name__ == '__main__':
    # 导入inference_runner以获取路径配置（使用多种导入方式）
    try:
        from server.inference_runner import (
            PROJECT_ROOT,
            GENERATE_SCRIPT,
            LYRIC_FILE_PATH,
            OUTPUT_DIR,
            AUDIO_OUTPUT_DIR
        )
    except ImportError:
        try:
            from .inference_runner import (
                PROJECT_ROOT,
                GENERATE_SCRIPT,
                LYRIC_FILE_PATH,
                OUTPUT_DIR,
                AUDIO_OUTPUT_DIR
            )
        except ImportError:
            from inference_runner import (
                PROJECT_ROOT,
                GENERATE_SCRIPT,
                LYRIC_FILE_PATH,
                OUTPUT_DIR,
                AUDIO_OUTPUT_DIR
            )
    
    # 确保必要的目录存在（使用绝对路径）
    sample_dir = os.path.join(PROJECT_ROOT, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    # 确保lyric文件目录存在
    lyric_dir = os.path.dirname(LYRIC_FILE_PATH)
    os.makedirs(lyric_dir, exist_ok=True)
    
    # 注册关闭钩子
    atexit.register(shutdown)
    
    safe_print("🎵 音乐生成服务器启动中...")
    safe_print(f"项目根目录: {PROJECT_ROOT}")
    safe_print(f"generate.sh脚本: {GENERATE_SCRIPT}")
    safe_print(f"lyric文件路径: {LYRIC_FILE_PATH}")
    safe_print(f"输出目录: {OUTPUT_DIR}")
    safe_print(f"音频输出目录: {AUDIO_OUTPUT_DIR}")
    
    # 验证路由注册
    safe_print("\n已注册的路由:")
    for rule in app.url_map.iter_rules():
        if rule.rule.startswith('/api/'):
            safe_print(f"  {rule.rule} [{', '.join([m for m in rule.methods if m != 'HEAD' and m != 'OPTIONS'])}]")
    
    # 获取本机IP地址
    import socket
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
        safe_print(f"\n服务器IP地址: {ip_address}")
    except:
        safe_print("\n无法获取IP地址")
    
    safe_print("\n服务器已就绪")
    safe_print("按 Ctrl+C 退出服务器\n")
    
    try:
        # 使用socketio.run启动服务器
        socketio.run(app, host='0.0.0.0', port=6006, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        safe_print("接收到中断信号，正在关闭服务器...")
        shutdown()

