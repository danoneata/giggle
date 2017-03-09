from typing import (
    Any,
    List,
)


class Recommender:

    def predict(self, user_id: int) -> List[Any]:
        return []


def load_model():
    return Recommender()
