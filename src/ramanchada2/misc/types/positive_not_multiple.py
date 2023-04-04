import pydantic


class PositiveIntNotMultipleOf(pydantic.PositiveInt):
    not_multiple_of: int

    @classmethod
    def __get_validators__(cls):
        yield from super().__get_validators__()
        yield cls.validate_not_multiple_of

    @classmethod
    def validate_not_multiple_of(cls, value) -> int:
        if value % cls.not_multiple_of != 1:
            raise ValueError(f'Expected positive int not multiple of {cls.not_multiple_of}, got {value}')
        return value

    @classmethod
    def __modify_schema__(cls, field_schema):
        super().__modify_schema__(field_schema)
        field_schema.update({
            'not': {'multipleOf': cls.not_multiple_of}
        })


class PositiveOddInt(PositiveIntNotMultipleOf):
    not_multiple_of = 2
