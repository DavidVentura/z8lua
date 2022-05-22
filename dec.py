# Parser adapted to 5.2 from https://openpunk.com/pages/lua-bytecode-parser/
# while looking at reference from https://blog.tst.sh/lua-5-2-5-3-bytecode-reference-incomplete/

'''
    Luac.py

    A Lua5.1 cross-platform bytecode deserializer. This module pulls int and size_t sizes from the
    chunk header, meaning it should be able to deserialize lua bytecode dumps from most platforms,
    regardless of the host machine.

    For details on the Lua5.1 bytecode format, I read [this PDF](https://archive.org/download/a-no-frills-intro-to-lua-5.1-vm-instructions/a-no-frills-intro-to-lua-5.1-vm-instructions_archive.torrent)
    as well as read the lundump.c source file from the Lua5.1 source.
'''

import struct
import array
from enum import IntEnum, Enum, auto

class InstructionType(Enum):
    ABC = auto(),
    ABx = auto(),
    AsBx = auto()
    Ax = auto()

class ConstType(IntEnum):
    NIL     = 0,
    BOOL    = 1,
    NUMBER  = 3,
    STRING  = 4,

class Instruction:
    def __init__(self, type: InstructionType, name: str) -> None:
        self.type = type
        self.name = name
        self.opcode: int = None
        self.A: int = None
        self.B: int = None
        self.C: int = None
        self.line: int = -1

    def toString(self, chunk: 'Chunk'):
        _s = str(self)
        # GETTABUP => a = b[c]; 
        if self.name == "GETTABUP":
            _c = chunk.constants[self.C*-1-1]
            _u = chunk.upvalues[self.B*-1-1]
            _s += f'; {_u.name} {_c}'

        # SETTABUP => a[b] = c;
        if self.name == "SETTABUP":
            _c = chunk.constants[self.B*-1-1]
            _u = chunk.upvalues[self.A*-1-1]
            _s += f'; {_u.name} {_c}'

        # LOADK => a = bx; 
        if self.name == "LOADK":
            const = chunk.constants[self.B]
            _s += f'; {const}'
        return _s

    def __str__(self):
        instr = "%10s" % self.name
        regs = ""

        if self.type == InstructionType.ABC:
            regs = "%d %d %d" % (self.A, self.B, self.C) 
        elif self.type == InstructionType.ABx or self.type == InstructionType.AsBx:
            regs = "%d %d" % (self.A, self.B)

        return "[%d] %s : %s" % (self.line, instr, regs)

class Constant:
    def __init__(self, type: ConstType, data) -> None:
        self.type = type
        self.data = data

    def __str__(self):
        printable_data = self.data
        if self.type == ConstType.NUMBER:
            #printable_data = float((self.data & 0xFFFF0000) >> 16) + ((self.data & 0xFFFF)/0xFFFF)
            _int = (self.data & 0xFFFF0000) >> 16
            _dec = (self.data & 0x0000FFFF)/0xFFFF
            printable_data = _int + _dec

        if self.type == ConstType.STRING:
            return f'"{printable_data}"'
        return str(printable_data)

    def toString(self):
        return str(self)

class Local:
    def __init__(self, name: str, start: int, end: int):
        self.name = name
        self.start = start
        self.end = end

    def __str__(self):
        return f'{self.name}\t{self.start}\t{self.end}'

class Upvalue:
    def __init__(self, idx: int, stack: int, register: int, name: str = '??'):
        self.idx = idx
        self.stack = stack
        self.register = register
        self.name = name

    def __str__(self):
        return f'{self.idx} {self.name} {self.stack} {self.register}'

class Chunk:
    def __init__(self) -> None:
        self.constants: list[Constant] = []
        self.instructions: list[Instruction] = []
        self.protos: list[Chunk] = []

        self.name: str = "Unnamed proto"
        self.frst_line: int = 0
        self.last_line: int = 0
        self.numUpvals: int = 0
        self.numParams: int = 0
        self.isVarg: bool = False
        self.maxStack: int = 0

        self.upvalues: list[Upvalue] = []
        self.locals: list[Local] = []

    def appendInstruction(self, instr: Instruction):
        self.instructions.append(instr)

    def appendConstant(self, const: Constant):
        self.constants.append(const)

    def appendProto(self, proto):
        self.protos.append(proto)

    def print(self):
        print("==== [[" + str(self.name) + "'s dissassembly]] ====")
        for i in range(len(self.instructions)):
            print("[%3d] %s" % (i, self.instructions[i].toString(self)))

        print("==== [[" + str(self.name) + "'s constants]] ====")
        for z in range(len(self.constants)):
            i = self.constants[z]
            print(str(z) + ": " + i.toString())

        print("==== [[" + str(self.name) + "'s locals]] ====")
        for i, l in enumerate(self.locals):
            print(f'\t{i}\t{l}')

        print("==== [[" + str(self.name) + "'s upvalues]] ====")
        for u in self.upvalues:
            print(u)

        print("==== [[" + str(self.name) + "'s protos]] ====")
        for z in self.protos:
            z.print()

instr_lookup_tbl = [
        Instruction(InstructionType.ABC, "MOVE"),
        Instruction(InstructionType.ABx, "LOADK"),
        Instruction(InstructionType.ABx, "LOADKX"),
        Instruction(InstructionType.ABC, "LOADBOOL"),
        Instruction(InstructionType.ABC, "LOADNIL"),
        Instruction(InstructionType.ABC, "GETUPVAL"),
        Instruction(InstructionType.ABC, "GETTABUP"),
        Instruction(InstructionType.ABC, "GETTABLE"),
        Instruction(InstructionType.ABC, "SETTABUP"),
        Instruction(InstructionType.ABC, "SETUPVAL"),
        Instruction(InstructionType.ABC, "SETTABLE"),
        Instruction(InstructionType.ABC, "NEWTABLE"),
        Instruction(InstructionType.ABC, "SELF"),
        Instruction(InstructionType.ABC, "ADD"),
        Instruction(InstructionType.ABC, "SUB"),
        Instruction(InstructionType.ABC, "MUL"),
        Instruction(InstructionType.ABC, "DIV"),
        Instruction(InstructionType.ABC, "MOD"),
        Instruction(InstructionType.ABC, "POW"),

        Instruction(InstructionType.ABC, "IDIV"),
        Instruction(InstructionType.ABC, "BAND"),
        Instruction(InstructionType.ABC, "BOR"),
        Instruction(InstructionType.ABC, "BXOR"),
        Instruction(InstructionType.ABC, "SHL"),
        Instruction(InstructionType.ABC, "SHR"),
        Instruction(InstructionType.ABC, "LSHR"),
        Instruction(InstructionType.ABC, "ROTL"),
        Instruction(InstructionType.ABC, "ROTR"),

        Instruction(InstructionType.ABC, "UNM"),
        Instruction(InstructionType.ABC, "BNOT"),
        Instruction(InstructionType.ABC, "NOT"),
        Instruction(InstructionType.ABC, "PEEK"),
        Instruction(InstructionType.ABC, "PEEK2"),
        Instruction(InstructionType.ABC, "PEEK4"),

        Instruction(InstructionType.ABC, "LEN"),
        Instruction(InstructionType.ABC, "CONCAT"),
        Instruction(InstructionType.AsBx, "JMP"),
        Instruction(InstructionType.ABC, "EQ"),
        Instruction(InstructionType.ABC, "LT"),
        Instruction(InstructionType.ABC, "LE"),
        Instruction(InstructionType.ABC, "TEST"),
        Instruction(InstructionType.ABC, "TESTSET"),
        Instruction(InstructionType.ABC, "CALL"),
        Instruction(InstructionType.ABC, "TAILCALL"),
        Instruction(InstructionType.ABC, "RETURN"),
        Instruction(InstructionType.AsBx, "FORLOOP"),
        Instruction(InstructionType.AsBx, "FORPREP"),
        Instruction(InstructionType.ABC, "TFORCALL"),
        Instruction(InstructionType.ABC, "TFORLOOP"),
        Instruction(InstructionType.ABC, "SETLIST"),
        Instruction(InstructionType.ABx, "CLOSURE"),
        Instruction(InstructionType.ABC, "VARARG"),
        Instruction(InstructionType.Ax, "EXTRAARG"),
        ]

# at [p]osition, with [s]ize of bits
def _get_bits(num, p, s):
    # convert number into binary first 
    binary = bin(num) 

    # remove first two characters 
    binary = binary[2:] 

    # fill in missing bits
    for i in range(32 - len(binary)):
        binary = '0' + binary

    start = len(binary) - (p+s)
    end = len(binary) - p

    # extract k  bit sub-string 
    kBitSubStr = binary[start : end]

    # convert extracted sub-string into decimal again 
    return (int(kBitSubStr,2))

class LuaUndump:
    def __init__(self):
        self.rootChunk: Chunk = None
        self.index = 0

    @staticmethod
    def dis_chunk(chunk: Chunk):
        chunk.print()
    
    def loadBlock(self, sz) -> bytearray:
        if self.index + sz > len(self.bytecode):
            raise Exception("Malformed bytecode!")

        temp = bytearray(self.bytecode[self.index:self.index+sz])
        # print(f"bytecode range for block of size {sz} is {['{:02x}'.format(x) for x in temp]}")
        self.index = self.index + sz
        return temp

    def get_byte(self) -> int:
        return self.loadBlock(1)[0]

    def get_int16(self) -> int:
        if (self.big_endian):
            return int.from_bytes(self.loadBlock(2), byteorder='big', signed=False)
        else:
            return int.from_bytes(self.loadBlock(2), byteorder='little', signed=False)

    def get_int32(self) -> int:
        if (self.big_endian):
            return int.from_bytes(self.loadBlock(4), byteorder='big', signed=False)
        else:
            return int.from_bytes(self.loadBlock(4), byteorder='little', signed=False)

    def get_int(self) -> int:
        if (self.big_endian):
            return int.from_bytes(self.loadBlock(self.int_size), byteorder='big', signed=False)
        else:
            return int.from_bytes(self.loadBlock(self.int_size), byteorder='little', signed=False)

    def get_size_t(self) -> int:
        if (self.big_endian):
            return int.from_bytes(self.loadBlock(self.size_t), byteorder='big', signed=False)
        else:
            return int.from_bytes(self.loadBlock(self.size_t), byteorder='little', signed=False)

    def get_string(self, size) -> str:
        if (size == None):
            size = self.get_size_t()
            if (size == 0):
                print('string is size 0')
                return ""

        return "".join(chr(x) for x in self.loadBlock(size))

    def decode_chunk(self):
        chunk = Chunk()

        chunk.frst_line = self.get_int32()
        chunk.last_line = self.get_int32()

        chunk.numParams = self.get_byte()
        chunk.isVarg = (self.get_byte() != 0)
        chunk.numUpvals = self.get_byte()

        if chunk.frst_line == 0:
            chunk.name = f"main<{chunk.frst_line},{chunk.last_line}>"
        elif (not chunk.name == ""):
            chunk.name = f"??<{chunk.frst_line},{chunk.last_line}>"

        # parse instructions
        num = self.get_int()
        for i in range(num):
            data   = self.get_int32()
            opcode = _get_bits(data, 0, 6)
            template = instr_lookup_tbl[opcode]
            instruction = Instruction(template.type, template.name)

            # i read the lopcodes.h file to get these bit position and sizes.
            instruction.opcode = opcode
            instruction.A = _get_bits(data, 6, 8) # starts after POS_OP + SIZE_OP (6), with a size of 8

            if instruction.type == InstructionType.ABC:
                instruction.B = _get_bits(data, 23, 9) # starts after POS_C + SIZE_C (23), with a size of 9
                instruction.C = _get_bits(data, 14, 9) # starts after POS_A + SIZE_A (14), with a size of 9
                if instruction.B > 255:
                    instruction.B = 255 - instruction.B
                if instruction.C > 255:
                    instruction.C = 255 - instruction.C
            elif instruction.type == InstructionType.ABx:
                instruction.B = _get_bits(data, 14, 18) # starts after POS_A + SIZE_A (14), with a size of 18
            elif instruction.type == InstructionType.AsBx:
                instruction.B = _get_bits(data, 14, 18) - 131071 # Bx is now signed, so just sub half of the MAX_UINT for 18 bits

            chunk.appendInstruction(instruction)

        # get constants
        num = self.get_int()
        for i in range(num):
            constant: Constant = None
            type = self.get_byte()

            if type == 0: #nil
                constant = Constant(ConstType.NIL, None)
            elif type == 1: # bool
                constant = Constant(ConstType.BOOL, (self.get_byte() != 0))
            elif type == 3: # number
                constant = Constant(ConstType.NUMBER, self.get_int32())
            elif type == 4: # string
                constant = Constant(ConstType.STRING, self.get_string(None)[:-1])
            else:
                raise Exception("Unknown Datatype! [%d]" % type)

            chunk.appendConstant(constant)

        # parse protos / "primitives"
        num = self.get_int()
        for i in range(num):
            chunk.appendProto(self.decode_chunk())

        # FIXME: order here was debug, locals, upvalues

        # upvalues
        num = self.get_int()
#        print(f'getting {num} upvalues')
        for i in range(num):
            stack = self.get_byte()
            register = self.get_byte()
            chunk.upvalues.append(Upvalue(i, stack, register))

        # line numbers
        source = self.get_string(None)[:-1]
        num = self.get_int()
        for i in range(num):
            line = self.get_int()
            chunk.instructions[i].line = line

        num = self.get_int()
        for i in range(num):
            localname = self.get_string(None)[:-1]
            startpc = self.get_int()
            endpc = self.get_int()
            chunk.locals.append(Local(localname, startpc, endpc))

        num = self.get_int()
        for i in range(num):
            upvalname = self.get_string(None)[:-1]
            chunk.upvalues[i].name = upvalname


        return chunk
        
    def decode_rawbytecode(self, rawbytecode):
        # bytecode sanity checks
        if not rawbytecode[0:4] == b'\x1bLua':
            raise Exception("Lua Bytecode expected!")

        bytecode = array.array('b', rawbytecode)
        return self.decode_bytecode(bytecode)

    def decode_bytecode(self, bytecode):
        self.bytecode   = bytecode

        # aligns index, skips header
        self.index = 4
        
        self.vm_version = self.get_byte()
        self.bytecode_format = self.get_byte()
        self.big_endian = (self.get_byte() == 0)
        self.int_size   = self.get_byte()
        self.size_t     = self.get_byte()
        self.instr_size = self.get_byte() # gets size of instructions
        self.l_number_size = self.get_byte() # size of lua_Number
        self.integral_flag = self.get_byte()

        assert self.get_byte() == 0x19
        assert self.get_byte() == 0x93
        assert self.get_byte() == ord('\r')
        assert self.get_byte() == ord('\n')
        assert self.get_byte() == 0x1a  # ?
        assert self.get_byte() == ord('\n')

        self.rootChunk = self.decode_chunk()
        return self.rootChunk
        
    def loadFile(self, luaCFile):
        with open(luaCFile, 'rb') as luac_file:
            bytecode = luac_file.read()
            return self.decode_rawbytecode(bytecode)

    def print_dissassembly(self):
        LuaUndump.dis_chunk(self.rootChunk)

    def find_localization_candidates(self):
        # recurse through all rootChunk.protos; find GETTABUP and SETTABUP to 
        self.rootChunk

lu = LuaUndump()
lu.loadFile('luac.out')
lu.print_dissassembly()

lu.find_localization_candidates()
