from pydantic import BaseModel, Field

class OrderSummary(BaseModel):
    product: str = Field(description="상품명")
    price: int = Field(description="가격")
    status: str = Field(description="배송 상태")