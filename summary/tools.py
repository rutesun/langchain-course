from langchain.tools import tool, BaseTool
from typing import List

# 1단계 도구: 이름으로 주문 ID 찾기
@tool
def get_order_id(user_name: str) -> str:
    """user_name을 입력받아 해당 사용자의 가장 최근 order_id를 반환합니다."""
    print(f"[Tool] get_order_id called for: {user_name}")
    # 가상의 데이터베이스 조회
    if user_name.lower() == "eden":
        return "ORD-9981"
    return "ORD-0000"

# 2단계 도구: 주문 ID로 상세 내역 조회하기
@tool
def get_order_details(order_id: str) -> str:
    """order_id를 입력받아 해당 주문의 상세 내역(상품명, 가격)을 반환합니다."""
    print(f"[Tool] get_order_details called for: {order_id}")
    
    if order_id == "ORD-9981":
        return "상품명: 무선 키보드, 가격: $150, 배송지: 서울"
    return "주문 내역 없음"

# 3단계 도구 (선택): 배송 상태 조회
@tool
def get_shipping_status(order_id: str) -> str:
    """order_id를 입력받아 현재 배송 상태를 반환합니다."""
    print(f"[Tool] get_shipping_status called for: {order_id}")
    
    if order_id == "ORD-9981":
        return "배송 중 (현재 위치: 대전 허브)"
    return "상태 알 수 없음"


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")