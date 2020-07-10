from itertools import zip_longest

import pytest


class Bits(bytes):
    CHUNK = 8

    def __new__(cls, n):
        return super().__new__(cls, cls.number_to_bits(n))

    @staticmethod
    def number_to_bits(n, size=CHUNK):
        length = Bits.how_many_bytes(n, size=size)
        mask = 2 ** size - 1
        offset = lambda i: i * size
        bit_slice = lambda n, i: (n & (mask << offset(i))) >> offset(i)

        return (bit_slice(n, i) for i in range(length))

    def to_number(self):
        return sum(byte_ * (2 ** (8 * i))
                   for i, byte_ in enumerate(super().__iter__()))

    @staticmethod
    def how_many_bytes(value, size=CHUNK):
        l = 1
        limit = (2 ** size) - 1
        while value > limit:
            value >>= size
            l += 1
        return l

    def __repr__(self):
        return f'{self.to_number()}'

    def __iter__(self):
        if self == bytes([0]):
            yield 0
            return

        size = self.CHUNK
        last_byte = len(self) - 1

        for byte_number, value in enumerate(super().__iter__()):
            count = 0

            while value:
                yield value & 1
                value >>= 1
                count += 1

            if byte_number < last_byte:
                yield from (0 for _ in range(size - count))

    def __reversed__(self):
        list_ = list(self)
        for n in reversed(list_):
            yield n

    def __lshift__(self, other):
        return Bits(self.to_number() << other.to_number())


class Byte(Bits):
    SIZE = 1

    def __new__(cls, n):
        if n >= cls.upper_bound():
            raise ValueError(f'{n}. Byte must be in range(0, {cls.upper_bound()})')

        return super().__new__(cls, n)

    @classmethod
    def upper_bound(cls):
        return 2 ** (cls.SIZE * cls.CHUNK)


class Word(Byte):
    SIZE = 2


class Tribyte(Byte):
    SIZE = 3


class DoubleWord(Byte):
    SIZE = 4


def fulladder(a, b, cin):
    sum_ = (a ^ b) ^ cin
    cout = a & b | cin & (a ^ b)
    return sum_, cout


def adder(n, m):
    acc = cout = cin = 0

    for i, (a, b) in enumerate(zip_longest(n, m, fillvalue=0)):
        sum_, cout = fulladder(a, b, cin)

        acc |= sum_ << i
        cin = cout

    acc |= cout << (i + 1)

    return acc


def subtractor(n, m):
    acc = 0
    cout = inverter = 1

    for i, (a, b) in enumerate(zip_longest(n, m, fillvalue=0)):
        b ^= inverter
        sum_, cout = fulladder(a, b, cout)

        acc |= sum_ << i

    return acc


def multiplier(n, m):
    acc = Bits(0)
    shifter = 0

    for bit in m:
        if bit == 1:
            acc = Bits(adder(acc, n << Bits(shifter)))
        shifter += 1

    return acc.to_number()


def divider(n, m):
    if m.to_number() == 0:
        raise ZeroDivisionError

    rem = quo = 0

    for bit in reversed(n):
        quo <<= 1
        rem <<= 1

        if bit:
            rem ^= (1 << 0)

        if rem >= m.to_number():
            rem = subtractor(Bits(rem), m)
            quo ^= (1 << 0)

    return quo, rem


def test_how_many_bytes():
    assert Bits.how_many_bytes(0) == 1
    assert Bits.how_many_bytes(1) == 1
    assert Bits.how_many_bytes(255) == 1
    assert Bits.how_many_bytes(256) == 2
    assert Bits.how_many_bytes(65535) == 2
    assert Bits.how_many_bytes(65536) == 3


def test_number_to_bits():
    assert list(Bits.number_to_bits(0)) == [0]
    assert list(Bits.number_to_bits(1)) == [1]
    assert list(Bits.number_to_bits(255)) == [255]
    assert list(Bits.number_to_bits(256)) == [0, 1]
    assert list(Bits.number_to_bits(65535)) == [255, 255]
    assert list(Bits.number_to_bits(65536)) == [0, 0, 1]


def test_bits_big_endian_layout():
    assert Bits(1) == bytes([1])
    assert Bits(255) == bytes([255])
    assert Bits(256) == bytes([0, 1])
    assert Bits(65535) == bytes([255, 255])
    assert Bits(65536) == bytes([0, 0, 1])
    assert Bits(131_070) == bytes([0xfe, 0xff, 0x1])


def test_bits_iter():
    assert list(Bits(0)) == [0]
    assert list(Bits(1)) == [1]
    assert list(Bits(2)) == [0, 1]
    assert list(Bits(128)) == [0, 0, 0, 0, 0, 0, 0, 1]
    assert list(Bits(255)) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert list(Bits(257)) == [1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert list(Bits(256)) == [0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert list(Bits(131_070)) == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_reversed_bits():
    assert list(reversed(list(Bits(0)))) == [0]
    assert list(reversed(list(Bits(1)))) == [1]
    assert list(reversed(list(Bits(2)))) == [1, 0]
    assert list(reversed(list(Bits(128)))) == [1, 0, 0, 0, 0, 0, 0, 0]
    assert list(reversed(list(Bits(255)))) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert list(reversed(list(Bits(257)))) == [1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert list(reversed(list(Bits(256)))) == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert list(reversed(list(Bits(131_070)))) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]


def test_bits_to_number():
    assert Bits(257).to_number() == 257


def test_bits_shift_left():
    assert Bits(1) << Bits(1) == Bits(2)


def test_adder():
    assert adder(Bits(0), Bits(0)) == 0
    assert adder(Bits(1), Bits(0)) == 1
    assert adder(Bits(0), Bits(1)) == 1
    assert adder(Bits(1), Bits(1)) == 2
    assert adder(Bits(5), Bits(12)) == 17
    assert adder(Bits(255), Bits(1)) == 256


def test_subtractor():
    assert subtractor(Bits(0), Bits(0)) == 0
    assert subtractor(Bits(1), Bits(1)) == 0
    assert subtractor(Bits(1), Bits(0)) == 1
    assert subtractor(Bits(10), Bits(5)) == 5
    assert subtractor(Bits(255), Bits(1)) == 254
    assert subtractor(Bits(256), Bits(1)) == 255


def test_multiplier():
    assert multiplier(Bits(0), Bits(0)) == 0
    assert multiplier(Bits(1), Bits(0)) == 0
    assert multiplier(Bits(0), Bits(1)) == 0
    assert multiplier(Bits(1), Bits(1)) == 1
    assert multiplier(Bits(2), Bits(3)) == 6
    assert multiplier(Bits(127), Bits(2)) == 254


def test_divider():
    with pytest.raises(ZeroDivisionError):
        divider(Bits(1), Bits(0))

    assert divider(Bits(0), Bits(1)) == (0, 0)
    assert divider(Bits(1), Bits(1)) == (1, 0)
    assert divider(Bits(6), Bits(3)) == (2, 0)
    assert divider(Bits(11), Bits(3)) == (3, 2)
    assert divider(Bits(18), Bits(5)) == (3, 3)
    assert divider(Bits(42), Bits(6)) == (7, 0)
    assert divider(Bits(254), Bits(2)) == (127, 0)
    assert divider(Bits(400), Bits(10)) == (40, 0)
    assert divider(Bits(512), Bits(10)) == (51, 2)
    assert divider(Bits(513), Bits(4)) == (128, 1)
    assert divider(Bits(512), Bits(2)) == (256, 0)
    assert divider(Bits(131_072), Bits(2)) == (65_536, 0)
    assert divider(Bits(33_554_432), Bits(2)) == (16_777_216, 0)
