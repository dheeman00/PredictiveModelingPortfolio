from typing import List

_value_list = ()  # Using a static set of values for the entire problem. This is problem specific.
_class_list = ()  # Using a static set of classes for the entire problem. This is problem specific.

#setting the values to List
def set_value_list(values: List[int]):
    global _value_list
    _value_list = tuple(values)

#setting the sample class as List
def set_class_list(classes: List[int]):
    global _class_list
    _class_list = tuple(classes)
