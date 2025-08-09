from dataclasses import dataclass

@dataclass
class McpError(Exception):
    code: str
    message: str
    hint: str | None = None

    def to_json(self):
        out = {"code": self.code, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        return out
