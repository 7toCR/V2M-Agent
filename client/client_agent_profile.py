"""
ClientAgent Profile 鏂囦欢

瀹氫箟ClientAgent鐨勮鑹层€佸伐鍏枫€佺煡璇嗐€佺害鏉熷拰鏈€浣冲疄璺点€?
璇ユ枃浠朵綅浜巆lient鐩綍涓嬶紝渚汣lientAgent浣跨敤銆?
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
        NETWORK_COMMUNICATION = "缃戠粶閫氫俊"
        FILE_TRANSFER = "鏂囦欢浼犺緭"
        SSH_MANAGEMENT = "SSH绠＄悊"
        TASK_COORDINATION = "浠诲姟鍗忚皟"
        OTHER = "鍏朵粬"

    class KnowledgeDomain(str, Enum):
        CLIENT_SERVER = "瀹㈡埛绔?鏈嶅姟鍣ㄩ€氫俊"
        NETWORK_PROTOCOLS = "缃戠粶鍗忚"
        FILE_HANDLING = "鏂囦欢澶勭悊"
        SECURITY = "瀹夊叏閫氫俊"
        ERROR_HANDLING = "閿欒澶勭悊"
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

# ClientAgent 瑙掕壊瀹氫箟
CLIENT_AGENT_ROLE = AgentRole(
    name="闊充箰鐢熸垚瀹㈡埛绔崗璋冧笓瀹?,
    description="璐熻矗涓庨煶涔愮敓鎴愭湇鍔″櫒閫氫俊锛岀鐞嗘枃浠朵笂浼犮€佷换鍔℃彁浜ゅ拰缁撴灉涓嬭浇锛屾敮鎸佽嚜鍔⊿SH闅ч亾寤虹珛",
    responsibilities=[
        "寤虹珛涓庨煶涔愮敓鎴愭湇鍔″櫒鐨勮繛鎺?,
        "涓婁紶lyric.jsonl鏂囦欢鍒版湇鍔″櫒",
        "鎻愪氦闊充箰鐢熸垚浠诲姟骞剁洃鎺х姸鎬?,
        "閫氳繃WebSocket鎺ユ敹瀹炴椂浠诲姟鏇存柊",
        "涓嬭浇鐢熸垚鐨勯煶棰戞枃浠跺埌鏈湴"
    ],
    expertise=[
        "瀹㈡埛绔?鏈嶅姟鍣ㄩ€氫俊",
        "鏂囦欢浼犺緭绠＄悊",
        "SSH闅ч亾鎶€鏈?,
        "WebSocket瀹炴椂閫氫俊",
        "閿欒鎭㈠澶勭悊"
    ]
)

# ClientAgent 宸ュ叿瀹氫箟
CLIENT_AGENT_TOOLS = [
    AgentTool(
        name="connect_to_server",
        function_signature="async def _connect_to_server_node(self, state: 'ClientAgent.Graph') -> Dict[str, Any]",
        description="寤虹珛涓庨煶涔愮敓鎴愭湇鍔″櫒鐨勮繛鎺ワ紝鏀寔鑷姩SSH闅ч亾",
        parameters=[
            {"name": "state", "type": "ClientAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "server_url", "type": "str", "description": "鏈嶅姟鍣║RL"},
            {"name": "auto_ssh", "type": "bool", "description": "鏄惁鑷姩寤虹珛SSH闅ч亾"}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈杩炴帴鐘舵€佷俊鎭?,
        category=ToolCategory.NETWORK_COMMUNICATION,
        usage_example="杩炴帴鍒癶ttp://localhost:6006鏈嶅姟鍣?,
        dependencies=["ssh_manager", "socketio_client"]
    ),
    AgentTool(
        name="upload_lyric_file",
        function_signature="async def _upload_lyric_file_node(self, state: 'ClientAgent.Graph') -> Dict[str, Any]",
        description="涓婁紶lyric.jsonl鏂囦欢鍒版湇鍔″櫒",
        parameters=[
            {"name": "state", "type": "ClientAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "lyric_file", "type": "str", "description": "lyric鏂囦欢璺緞"},
            {"name": "output_dir", "type": "str", "description": "杈撳嚭鐩綍璺緞"}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈涓婁紶缁撴灉",
        category=ToolCategory.FILE_TRANSFER,
        usage_example="涓婁紶lyric.jsonl鏂囦欢鍒版湇鍔″櫒",
        dependencies=["file_handler", "http_client"]
    ),
    AgentTool(
        name="start_inference_task",
        function_signature="async def _start_inference_task_node(self, state: 'ClientAgent.Graph') -> Dict[str, Any]",
        description="鍚姩闊充箰鐢熸垚鎺ㄧ悊浠诲姟",
        parameters=[
            {"name": "state", "type": "ClientAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "task_data", "type": "dict", "description": "浠诲姟鏁版嵁"}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈浠诲姟ID鍜岀姸鎬?,
        category=ToolCategory.TASK_COORDINATION,
        usage_example="鎻愪氦闊充箰鐢熸垚浠诲姟鍒版湇鍔″櫒",
        dependencies=["task_manager", "websocket_client"]
    ),
    AgentTool(
        name="download_audio_files",
        function_signature="async def _download_audio_files_node(self, state: 'ClientAgent.Graph') -> Dict[str, Any]",
        description="涓嬭浇鐢熸垚鐨勯煶棰戞枃浠跺埌鏈湴",
        parameters=[
            {"name": "state", "type": "ClientAgent.Graph", "description": "褰撳墠鐘舵€佸浘鐘舵€?},
            {"name": "task_id", "type": "str", "description": "浠诲姟ID"},
            {"name": "output_dir", "type": "str", "description": "杈撳嚭鐩綍璺緞"}
        ],
        returns="鏇存柊鍚庣殑鐘舵€佸瓧鍏革紝鍖呭惈涓嬭浇缁撴灉",
        category=ToolCategory.FILE_TRANSFER,
        usage_example="涓嬭浇.flac闊抽鏂囦欢鍒版湰鍦扮洰褰?,
        dependencies=["file_handler", "http_client"]
    )
]

# ClientAgent 鐭ヨ瘑瀹氫箟
CLIENT_AGENT_KNOWLEDGE = [
    AgentKnowledge(
        domain=KnowledgeDomain.CLIENT_SERVER,
        concepts=[
            "REST API閫氫俊鍗忚",
            "WebSocket瀹炴椂鏇存柊",
            "HTTP鏂囦欢浼犺緭",
            "浠诲姟鐘舵€佺鐞?,
            "閿欒澶勭悊鍜岄噸璇?
        ],
        rules=[
            "浣跨敤姝ｇ‘鐨凙PI绔偣杩涜閫氫俊",
            "瀹炴椂鐩戞帶浠诲姟鐘舵€佸彉鍖?,
            "澶勭悊杩炴帴涓柇鍜岄噸杩?,
            "绠＄悊鏂囦欢浼犺緭鐨勫畬鏁存€?,
            "浼樺寲閫氫俊鎬ц兘鍜屽彲闈犳€?
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.NETWORK_PROTOCOLS,
        concepts=[
            "SSH闅ч亾寤虹珛鍜岀淮鎶?,
            "绔彛杞彂鎶€鏈?,
            "缃戠粶杩炴帴浼樺寲",
            "瀹夊叏閫氫俊鍗忚",
            "鎬ц兘鐩戞帶鍜屽垎鏋?
        ],
        rules=[
            "鑷姩寤虹珛鍜岀淮鎶SH闅ч亾",
            "绠＄悊绔彛杞彂閰嶇疆",
            "浼樺寲缃戠粶杩炴帴鎬ц兘",
            "纭繚閫氫俊瀹夊叏鎬?,
            "鐩戞帶缃戠粶璧勬簮浣跨敤"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.FILE_HANDLING,
        concepts=[
            "鏂囦欢涓婁紶涓嬭浇鎶€鏈?,
            "鏍煎紡楠岃瘉鍜屽鐞?,
            "瀛樺偍绠＄悊绛栫暐",
            "瀹屾暣鎬ф鏌ユ柟娉?,
            "鎬ц兘浼樺寲鎶€宸?
        ],
        rules=[
            "楠岃瘉鏂囦欢鏍煎紡鍜屽畬鏁存€?,
            "浼樺寲鏂囦欢浼犺緭鎬ц兘",
            "绠＄悊鏈湴瀛樺偍绌洪棿",
            "澶勭悊鏂囦欢浼犺緭閿欒",
            "纭繚鏁版嵁涓€鑷存€?
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.SECURITY,
        concepts=[
            "SSH璁よ瘉鍜屾巿鏉?,
            "鏁版嵁浼犺緭鍔犲瘑",
            "璁块棶鎺у埗绠＄悊",
            "瀹夊叏瀹¤鏃ュ織",
            "婕忔礊闃茶寖鎺柦"
        ],
        rules=[
            "瀹夊叏瀛樺偍鍜屼娇鐢⊿SH鍑瘉",
            "鍔犲瘑鏁忔劅鏁版嵁浼犺緭",
            "瀹炴柦璁块棶鎺у埗绛栫暐",
            "璁板綍瀹夊叏瀹¤鏃ュ織",
            "闃茶寖瀹夊叏婕忔礊"
        ]
    ),
    AgentKnowledge(
        domain=KnowledgeDomain.ERROR_HANDLING,
        concepts=[
            "杩炴帴閿欒鎭㈠",
            "浼犺緭閿欒澶勭悊",
            "浠诲姟澶辫触閲嶈瘯",
            "璧勬簮娓呯悊绠＄悊",
            "鐢ㄦ埛鍙嶉鎻愪緵"
        ],
        rules=[
            "鑷姩澶勭悊杩炴帴閿欒鍜岄噸杩?,
            "鎭㈠鏂囦欢浼犺緭涓柇",
            "閲嶈瘯澶辫触鐨勪换鍔?,
            "娓呯悊涓存椂璧勬簮",
            "鎻愪緵娓呮櫚鐨勯敊璇俊鎭?
        ]
    )
]

# ClientAgent 绾︽潫鏉′欢
CLIENT_AGENT_CONSTRAINTS = [
    "蹇呴』浣跨敤瀹夊叏鐨凷SH闅ч亾杩炴帴",
    "楠岃瘉鏂囦欢鏍煎紡鍜屽畬鏁存€у悗鍐嶄笂浼?,
    "瀹炴椂鐩戞帶浠诲姟鐘舵€佸拰杩涘害",
    "澶勭悊缃戠粶杩炴帴涓柇鍜岄敊璇?,
    "纭繚鏂囦欢浼犺緭鐨勫畬鏁存€у拰瀹夊叏鎬?,
    "涓嶅緱娉勯湶SSH璁よ瘉淇℃伅",
    "闇€瑕佹彁渚涜缁嗙殑鎵ц鏃ュ織"
]

# ClientAgent 鏈€浣冲疄璺?
CLIENT_AGENT_BEST_PRACTICES = [
    "鍏堝缓绔嬬ǔ瀹氱殑SSH闅ч亾杩炴帴",
    "楠岃瘉杈撳叆鏂囦欢鐨勬牸寮忓拰鍐呭",
    "瀹炴椂鐩戞帶浠诲姟鎵ц鐘舵€?,
    "澶勭悊缃戠粶閿欒鍜岃嚜鍔ㄩ噸璇?,
    "浼樺寲鏂囦欢浼犺緭鎬ц兘鍜屽彲闈犳€?,
    "鎻愪緵璇︾粏鐨勮繘搴﹀弽棣?,
    "娓呯悊涓存椂鏂囦欢鍜岃祫婧?
]

# ClientAgent Profile
CLIENT_AGENT_PROFILE = AgentProfile(
    agent_id="client_agent",
    role=CLIENT_AGENT_ROLE,
    tools=CLIENT_AGENT_TOOLS,
    knowledge=CLIENT_AGENT_KNOWLEDGE,
    constraints=CLIENT_AGENT_CONSTRAINTS,
    best_practices=CLIENT_AGENT_BEST_PRACTICES
)

# 瀵煎嚭鍑芥暟
def get_client_agent_profile() -> AgentProfile:
    """鑾峰彇ClientAgent鐨勯厤缃俊鎭?""
    return CLIENT_AGENT_PROFILE

def print_client_agent_profile():
    """Print ClientAgent configuration in English"""
    profile = CLIENT_AGENT_PROFILE
    print("=" * 80)
    print("ClientAgent Profile")
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
    print_client_agent_profile()
