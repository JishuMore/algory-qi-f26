"""Minimal Prosperity-compatible datamodel (enough for local backtesting)."""
from typing import Dict, List

Symbol = str
Product = str
Position = int


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"


class OrderDepth:
    def __init__(self) -> None:
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: str = '', seller: str = '',
                 timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.buyer} << {self.seller}, {self.price}, {self.quantity}, {self.timestamp})"


class Observation:
    def __init__(self) -> None:
        self.plainValueObservations: Dict[Product, int] = {}
        self.conversionObservations: Dict[Product, object] = {}


class TradingState:
    def __init__(self, traderData: str, timestamp: int, listings: Dict, order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position], observations: Observation) -> None:
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
