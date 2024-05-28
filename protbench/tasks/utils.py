from typing import Any


def validate_type(attribute_name: str, value: Any, valid_type: Any) -> None:
    """Validate that the type of an attribute is the expected one.

    Args:
        attribute_name (str): name of the attribute. Used for error message.
        value (Any): attribute value.
        valid_type (Any): expected type of the attribute.

    Raises:
        TypeError: if the type of the attribute is not the expected one.
    """
    if not isinstance(value, valid_type):
        raise TypeError(
            f"Expected {attribute_name} to be {valid_type} but "
            f"got {type(value)} instead."
        )
