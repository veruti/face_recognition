from pydantic import BaseModel

# TODO: Maybe make BBox model as parent model
class Face(BaseModel):
    x_min: int
    x_max: int

    y_max: int
    y_min: int

    def get_min_point(self) -> tuple[int]:
        return (self.x_min, self.y_min)

    def get_max_point(self) -> tuple[int]:
        return (self.x_max, self.y_max)
