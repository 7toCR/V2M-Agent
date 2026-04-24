"""
ServerAgent Profile 鏂囦欢

瀹氫箟ServerAgent鐨勮鑹层€佸伐鍏枫€佺煡璇嗐€佺害鏉熷拰鏈€浣冲疄璺点€?
璇ユ枃浠朵綅浜巗erver鐩綍涓嬶紝渚汼erverAgent浣跨敤銆?
"""

import os
import sys
from typing import List, Dict, Any, Optional

# 瀵煎叆鍩虹绫诲瀷
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from task.task_profile import AgentProfile, AgentRole, AgentTool, AgentKnowledge, ToolCategory, KnowledgeDomain
except ImportError:
    # 濡傛灉娌℃湁task妯″潡锛屼娇鐢ㄦ湰鍦板畾涔?
    from dataclasses import dataclass
    from enum import Enum

    class ToolCategory(str, Enum):
        API_MANAGEMENT = "API绠＄悊"
        TASK_ORCHESTRATION = "浠诲姟缂栨帓"
        REQUEST_HANDLING = "璇锋眰澶勭悊"
        RESOURCE_MANAGEMENT = "璧勬簮绠＄悊"
        OTHER = "鍏朵粬"

    class KnowledgeDomain(str, Enum):
        SERVER_ARCHITECTURE = "鏈嶅姟鍣ㄦ灦鏋?
        API_DESIGN = "API璁捐"
        TASK_SCHEDULING = "浠诲姟璋冨害"
        CONCURRENT_PROCESSING = "骞跺彂澶勭悊"
        PERFORMANCE_OPTIMIZATION = "鎬ц兘浼樺寲"
        OTHER = "鍏朵粬"

    @dataclass
    class AgentRole:
        name: str
        description: str
        responsibilities: List[str]
        expertise: List[str]

    @dataclass
    class AgentTool:
        name: str
        function_signature: str
        description: str
        parameters: List[Dict[str, str]]
        returns: str
        category: ToolCategory
        usage_example: Optional[str] = None
        dependencies: List[str] = None

        def __post_init__(self):
            if self.dependencies is None:
                self.dependencies = []

    @dataclass
    class AgentKnowledge:
        domain: KnowledgeDomain
        concepts: List[str]
        rules: List[str]

    @dataclass
    class AgentProfile:
        agent_id: str
        role: AgentRole
        tools: List[AgentTool]
        knowledge: List[AgentKnowledge]
        constraints: List[str]
        best_practices: List[str]

# ServerAgent 瑙掕壊瀹氫箟
SERVER_AGENT_ROLE = AgentRole(
    name="闊充箰鐢熸垚鏈嶅姟鍣ㄧ鐞嗕笓瀹?,
    description="璐熻矗绠＄悊闊充箰鐢熸垚鏈嶅姟鍣紝澶勭悊瀹㈡埛绔姹傘€佺紪鎺掓帹鐞嗕换鍔°€佺洃鎺ф墽琛岀姸鎬侊紝鎻愪緵REST API鍜學ebSocket瀹炴椂閫氫俊",
    responsibilities=[
        "鎻愪緵REST API鎺ュ彛澶勭悊瀹㈡埛绔姹?,
        "绠＄悊闊充箰鐢熸垚浠诲姟鐨勮皟搴﹀拰鎵ц",
        "鐩戞帶浠诲姟鎵ц鐘舵€佸拰璧勬簮浣跨敤",
        "閫氳繃WebSocket鎻愪緵瀹炴椂鐘舵€佹洿鏂?,
        "澶勭悊鏂囦欢涓婁紶涓嬭浇鍜屽瓨鍌ㄧ鐞?
    ],
    expertise=[
        "鏈嶅姟鍣ㄦ灦鏋勮璁?,
        "API鎺ュ彛寮€鍙?,
        "浠诲姟璋冨害绠楁硶",
        "骞跺彂澶勭悊鎶€鏈?,
        "鎬ц兘浼樺寲绠＄悊"
    ]
)

# ServerAgent 宸ュ叿瀹氫箟
SERVER_AGENT_TOOLS = [
    AgentTool(
        name="handle_upload_request",
        function_signature="async def _handle_upload_request_node(self, state: 'ServerAgent.Graph') -> Dict[str, Any]",
        description="澶勭悊瀹㈡埛绔笂浼爈yric鏂囦欢鐨勮姹?,
        parameters=[
            {"name": "state", "type": "ServerAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "request_data", "type": "dict", "description": "璇锋眰鏁版嵁"},
            {"name": "client_info", "type": "dict", "description": "瀹㈡埛绔俊鎭?}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈澶勭悊缁撴灉",
        category=ToolCategory.REQUEST_HANDLING,
        usage_example="澶勭悊POST /api/upload_lyric璇锋眰",
        dependencies=["flask_server", "file_handler"]
    ),
    AgentTool(
        name="schedule_inference_task",
        function_signature="async def _schedule_inference_task_node(self, state: 'ServerAgent.Graph') -> Dict[str, Any]",
        description="璋冨害闊充箰鐢熸垚鎺ㄧ悊浠诲姟",
        parameters=[
            {"name": "state", "type": "ServerAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "task_data", "type": "dict", "description": "浠诲姟鏁版嵁"},
            {"name": "priority", "type": "int", "description": "浠诲姟浼樺厛绾?}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈璋冨害缁撴灉",
        category=ToolCategory.TASK_ORCHESTRATION,
        usage_example="璋冨害闊充箰鐢熸垚浠诲姟鍒版帹鐞嗛槦鍒?,
        dependencies=["task_manager", "inference_runner"]
    ),
    AgentTool(
        name="monitor_task_status",
        function_signature="async def _monitor_task_status_node(self, state: 'ServerAgent.Graph') -> Dict[str, Any]",
        description="鐩戞帶浠诲姟鎵ц鐘舵€佸拰杩涘害",
        parameters=[
            {"name": "state", "type": "ServerAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "task_id", "type": "str", "description": "浠诲姟ID"}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈鐩戞帶缁撴灉",
        category=ToolCategory.RESOURCE_MANAGEMENT,
        usage_example="鐩戞帶浠诲姟鎵ц杩涘害鍜岃祫婧愪娇鐢?,
        dependencies=["task_monitor", "resource_tracker"]
    ),
    AgentTool(
        name="broadcast_status_updates",
        function_signature="async def _broadcast_status_updates_node(self, state: 'ServerAgent.Graph') -> Dict[str, Any]",
        description="閫氳繃WebSocket骞挎挱浠诲姟鐘舵€佹洿鏂?,
        parameters=[
            {"name": "state", "type": "ServerAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "update_data", "type": "dict", "description": "鏇存柊鏁版嵁"},
            {"name": "client_ids", "type": "List[str]", "description": "瀹㈡埛绔疘D鍒楄〃"}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈骞挎挱缁撴灉",
        category=ToolCategory.API_MANAGEMENT,
        usage_example="閫氳繃WebSocket骞挎挱浠诲姟杩涘害鏇存柊",
        dependencies=["socketio_server", "websocket_manager"]
    )
]

# ServerAgent 鐭ヨ瘑瀹氫箟
SERVER_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.SERVER_ARCHITECTURE,
        concepts=[
            "Flask + Socket.IO鏈嶅姟鍣ㄦ灦鏋?,
            "REST API璁捐鍘熷垯",
            "WebSocket瀹炴椂閫氫俊",
            "骞跺彂璇锋眰澶勭悊",
            "璐熻浇鍧囪　绛栫暐"
        ],
        rules=[
            "鎻愪緵绋冲畾鍙潬鐨凙PI鎺ュ彛",
            "鏀寔瀹炴椂鐘舵€佹洿鏂?,
            "澶勭悊楂樺苟鍙戣姹?,
            "浼樺寲鏈嶅姟鍣ㄦ€ц兘",
            "纭繚鏈嶅姟鍙敤鎬?
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.API_DESIGN,
        concepts=[
            "RESTful API璁捐瑙勮寖",
            "璇锋眰鍝嶅簲鏍煎紡",
            "閿欒澶勭悊鏈哄埗",
            "瀹夊叏璁よ瘉鎺堟潈",
            "鐗堟湰绠＄悊绛栫暐"
        ],
        rules=[
            "閬靛惊RESTful璁捐鍘熷垯",
            "鎻愪緵娓呮櫚鐨凙PI鏂囨。",
            "缁熶竴閿欒澶勭悊鏍煎紡",
            "瀹炴柦瀹夊叏璁よ瘉鏈哄埗",
            "鏀寔API鐗堟湰绠＄悊"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.TASK_SCHEDULING,
        concepts=[
            "浠诲姟闃熷垪绠＄悊",
            "浼樺厛绾ц皟搴︾畻娉?,
            "骞跺彂鎵ц鎺у埗",
            "璧勬簮鍒嗛厤绛栫暐",
            "璐熻浇鍧囪　鎶€鏈?
        ],
        rules=[
            "鍚堢悊璋冨害浠诲姟鎵ц椤哄簭",
            "浼樺寲璧勬簮鍒嗛厤鍜屼娇鐢?,
            "鎺у埗骞跺彂鎵ц鏁伴噺",
            "澶勭悊浠诲姟浼樺厛绾у啿绐?,
            "骞宠　绯荤粺璐熻浇"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.CONCURRENT_PROCESSING,
        concepts=[
            "澶氱嚎绋?澶氳繘绋嬪鐞?,
            "寮傛鎵ц鎶€鏈?,
            "绾跨▼瀹夊叏璁捐",
            "閿佹満鍒剁鐞?,
            "鍐呭瓨鍏变韩绛栫暐"
        ],
        rules=[
            "瀹夊叏澶勭悊骞跺彂璇锋眰",
            "浼樺寲寮傛鎵ц鎬ц兘",
            "纭繚绾跨▼瀹夊叏鎬?,
            "鍚堢悊浣跨敤閿佹満鍒?,
            "绠＄悊鍏变韩璧勬簮"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.PERFORMANCE_OPTIMIZATION,
        concepts=[
            "鏈嶅姟鍣ㄦ€ц兘鐩戞帶",
            "璧勬簮浣跨敤鍒嗘瀽",
            "鐡堕璇嗗埆鎶€鏈?,
            "浼樺寲绛栫暐璁捐",
            "鎵╁睍鎬ц鍒?
        ],
        rules=[
            "瀹炴椂鐩戞帶鏈嶅姟鍣ㄦ€ц兘",
            "鍒嗘瀽璧勬簮浣跨敤鏁堢巼",
            "璇嗗埆鎬ц兘鐡堕",
            "璁捐浼樺寲绛栫暐",
            "瑙勫垝绯荤粺鎵╁睍"
        ]
    )
]

# ServerAgent 绾︽潫鏉′欢
SERVER_AGENT_CONSTRAINTS = [
    "蹇呴』鎻愪緵绋冲畾鍙潬鐨凙PI鎺ュ彛",
    "瀹炴椂鐩戞帶浠诲姟鎵ц鐘舵€?,
    "澶勭悊楂樺苟鍙戝鎴风璇锋眰",
    "纭繚浠诲姟璋冨害鐨勫叕骞虫€?,
    "涓嶅緱娉勯湶瀹㈡埛绔晱鎰熶俊鎭?,
    "闇€瑕佷紭鍖栨湇鍔″櫒鎬ц兘",
    "蹇呴』鎻愪緵璇︾粏鐨勬湇鍔℃棩蹇?
]

# ServerAgent 鏈€浣冲疄璺?
SERVER_AGENT_BEST_PRACTICES = [
    "璁捐娓呮櫚绋冲畾鐨凙PI鎺ュ彛",
    "浼樺寲浠诲姟璋冨害鍜岃祫婧愬垎閰?,
    "瀹炴椂鐩戞帶鏈嶅姟鍣ㄦ€ц兘",
    "澶勭悊骞跺彂璇锋眰鍜岄敊璇仮澶?,
    "鎻愪緵璇︾粏鐨勬墽琛屾棩蹇?,
    "浼樺寲WebSocket閫氫俊鎬ц兘",
    "瑙勫垝绯荤粺鎵╁睍鍜屽鐏?
]

# ServerAgent Profile
SERVER_AGENT_PROFILE = AgentProfile(
    agent_id="server_agent",
    role=SERVER_AGENT_ROLE,
    tools=SERVER_AGENT_TOOLS,
    knowledge=SERVER_AGENT_KNOWLEDGE,
    constraints=SERVER_AGENT_CONSTRAINTS,
    best_practices=SERVER_AGENT_BEST_PRACTICES
)

# 瀵煎嚭鍑芥暟
def get_server_agent_profile() -> AgentProfile:
    """鑾峰彇ServerAgent鐨勯厤缃俊鎭?""
    return SERVER_AGENT_PROFILE

def print_server_agent_profile():
    """Print ServerAgent configuration in English"""
    profile = SERVER_AGENT_PROFILE
    print("=" * 80)
    print("ServerAgent Profile")
    print("=" * 80)

    print(f"\nAgent ID: {profile.agent_id}")
    print(f"\nRole: {profile.role.name}")
    print(f"Role Description: {profile.role.description}")
    print(f"\nResponsibilities ({len(profile.role.responsibilities)}):")
    for i, resp in enumerate(profile.role.responsibilities, 1):
        print(f"  {i}. {resp}")
    print(f"\nExpertise ({len(profile.role.expertise)}):")
    for i, exp in enumerate(profile.role.expertise, 1):
        print(f"  {i}. {exp}")

    print(f"\nTools ({len(profile.tools)}):")
    for tool in profile.tools:
        print(f"  - {tool.name}: {tool.description}")
        if tool.dependencies:
            print(f"    Dependencies: {', '.join(tool.dependencies)}")

    print(f"\nKnowledge Domains ({len(profile.knowledge)}):")
    for knowledge in profile.knowledge:
        print(f"  - {knowledge.domain.value}:")
        print(f"    Concepts: {', '.join(knowledge.concepts[:3])}...")
        print(f"    Rules: {', '.join(knowledge.rules[:2])}...")

    print(f"\nConstraints ({len(profile.constraints)}):")
    for i, constraint in enumerate(profile.constraints, 1):
        print(f"  {i}. {constraint}")

    print(f"\nBest Practices ({len(profile.best_practices)}):")
    for i, practice in enumerate(profile.best_practices, 1):
        print(f"  {i}. {practice}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_server_agent_profile()
