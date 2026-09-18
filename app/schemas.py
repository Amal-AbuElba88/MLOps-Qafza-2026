"""Validated request and response contracts for the inference API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BRAZILIAN_STATES = {
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
}
PAYMENT_TYPES = {"boleto", "credit_card", "debit_card", "voucher"}


class OrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_state: str = Field(examples=["SP"])
    total_price: float = Field(ge=0, le=1_000_000, examples=[150.0])
    total_freight: float = Field(ge=0, le=1_000_000, examples=[20.0])
    num_items: int = Field(ge=1, le=10_000, examples=[2])
    num_unique_sellers: int = Field(ge=1, le=10_000, examples=[1])
    total_payment: float = Field(ge=0, le=2_000_000, examples=[170.0])
    max_installments: int = Field(ge=0, le=100, examples=[3])
    main_payment_type: str = Field(examples=["credit_card"])

    order_purchase_timestamp: Optional[datetime] = Field(
        None, examples=["2018-05-09T14:00:00Z"]
    )
    order_estimated_delivery_date: Optional[datetime] = Field(
        None, examples=["2018-05-19T14:00:00Z"]
    )
    purchase_year: Optional[int] = Field(None, ge=2016, le=2100)
    purchase_month: Optional[int] = Field(None, ge=1, le=12)
    purchase_dayofweek: Optional[int] = Field(None, ge=0, le=6)
    purchase_hour: Optional[int] = Field(None, ge=0, le=23)
    is_weekend: Optional[int] = Field(None, ge=0, le=1)
    estimated_delivery_days: Optional[float] = Field(None, ge=0, le=3650)
    freight_ratio: Optional[float] = Field(None, ge=0, le=1000)

    @field_validator("customer_state")
    @classmethod
    def validate_state(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in BRAZILIAN_STATES:
            raise ValueError("customer_state must be a valid Brazilian state code")
        return normalized

    @field_validator("main_payment_type")
    @classmethod
    def validate_payment_type(cls, value: str) -> str:
        if value not in PAYMENT_TYPES:
            raise ValueError(
                f"main_payment_type must be one of {sorted(PAYMENT_TYPES)}"
            )
        return value

    @model_validator(mode="after")
    def validate_feature_sources(self) -> "OrderInput":
        date_parts = (
            self.purchase_year,
            self.purchase_month,
            self.purchase_dayofweek,
            self.purchase_hour,
        )
        if self.order_purchase_timestamp is None and any(
            value is None for value in date_parts
        ):
            raise ValueError(
                "Provide order_purchase_timestamp or all purchase date features"
            )
        if (
            self.estimated_delivery_days is None
            and self.order_estimated_delivery_date is None
        ):
            raise ValueError(
                "Provide estimated_delivery_days or order_estimated_delivery_date"
            )
        return self


class PredictionResponse(BaseModel):
    request_id: str
    prediction: str
    is_late: int = Field(ge=0, le=1)
    probability: Optional[float] = Field(None, ge=0, le=1)
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int = Field(ge=1)


class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: str
    model_source: str
