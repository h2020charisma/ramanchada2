from pydantic import PositiveInt

# For more info check https://github.com/pydantic/pydantic/issues/10111
# PositiveOddInt = typing.Annotated[int,
#                                   annotated_types.Ge(0),
#                                   annotated_types.Not(annotated_types.MultipleOf(2))]

# FIXME This is a quickfix
PositiveOddInt = PositiveInt
