def add_no_think(s: str) -> str:
    if s.rstrip().endswith("/no_think"):
        return s
    return s.rstrip() + "\n/no_think"