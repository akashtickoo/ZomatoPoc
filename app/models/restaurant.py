"""Cleaned restaurant record (architecture §5.2)."""

from typing import Literal

from pydantic import BaseModel, Field

Budget = Literal["low", "medium", "high"]


class Restaurant(BaseModel):
    restaurant_id: str
    name: str
    address: str
    city: str = "Bangalore"
    location: str  # neighbourhood the restaurant is in
    areas: list[str]  # Zomato listing areas it appears under (listed_in(city)); can be several
    cuisines: list[str]
    cost_for_two: int | None = Field(None, gt=0)
    budget: Budget | None
    rating: float | None = Field(None, ge=0, le=5)
    votes: int = Field(ge=0)
    rest_type: list[str]
    online_order: bool
    book_table: bool
    dish_liked: list[str]
    listing_types: list[str]
    review_snippets: list[str]
