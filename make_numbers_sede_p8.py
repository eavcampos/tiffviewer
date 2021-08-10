def make_numbers_sede_p8(line, col, numLines, numColumns, p1Block: int):

        if p1Block == 5:
            temp=numLines
            numLines = numColumns
            numColums = temp
#pensar no restante da lógica       
        index = col * numLines + abs(line - oy)
        index += start_number
        return p1Block, index
