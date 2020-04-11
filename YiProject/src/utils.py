def writeFile(path, sentenceList):
    file_write_obj = open(path, 'a+')
    for var in sentenceList:
        write = ''
        sentence = var.sentence
        tabNum = var.tabNum
        for i in range(0, tabNum):
            write += '    '
        write += sentence
        file_write_obj.writelines(write)
        file_write_obj.write('\n')
        for j in range(0, var.blank):
            file_write_obj.write('\n')
    file_write_obj.close()


def copyFile(path, path1):

    fp = open(path, 'r')
    fp1 = open(path1, 'w')
    for i in fp:
        fp1.write(i)
    fp.close()
    fp1.close()
