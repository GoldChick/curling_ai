'''
编写基础策略,如占位、击飞、保护等
'''


def get_strategy_by_index(index: int) -> list:
    pass


def to_the_mid() -> list:
    return [3.0, 0.0, 0.0]


def protect_target(x, y) -> list:
    v0 = 3.613 - 0.12234*y + 1
    h0 = x - 2.375
    return [v0, h0, 0.0]


def dash_target(x, y) -> list:
    v0 = 3.613 - 0.12234*y + 1
    h0 = x - 2.375
    return [v0, h0, 0.0]
