import sys, re
with open('parametros') as f:
    linhas = f.readlines()[1:]
    con = 0
    linha_num = int(sys.argv[1]) - 1
    linha = linhas[linha_num]
    print("o argumento eh {}".format(linha_num))
    linha = linha[:-1] # tirar o \n
    linha = re.split(r'\t+', linha)

