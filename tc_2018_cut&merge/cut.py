from csv import DictWriter
import pandas as pd
import os

def cut_feature():
    os.chdir(r'..\data')
    ix = 0

    fo =  open('userFeature%s.csv' %ix, 'w')
    headers = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1', 'interest2',
    	'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',  'topic1', 'topic2', 'topic3', 'appIdInstall',
    	'appIdAction', 'ct', 'os', 'carrier', 'house']
    writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
    writer.writeheader()

    fi = open('userFeature.data', 'r')
    for line in fi :
        line = line.replace('\n', '').split('|')
        userFeature_dict = {}
        for each in line:
            each_list = each.split(' ')
            userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
        writer.writerow(userFeature_dict)
        ix = ix+1
        if ix % 400000==0:
            print(ix)
            #fo.close()
            fo = open('userFeature%s.csv' %ix, 'w')
            writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
            writer.writeheader()
    fo.close()
    fi.close()

def cut_train():
    df_train = pd.read_csv('../data/train.csv')

    ix = 0
    df_part = df_part = pd.DataFrame()
    for index,row in df_train.iterrows():
        #print(row['aid'])
        df_row = pd.DataFrame({'aid': [row['aid']],
                                'uid': [row['uid']],
                                'label': [row['label']]})
        df_part = pd.concat([df_part,df_row])
        ix = ix + 1
        if ix%400000  == 0:
            print(ix)
            df_part.to_csv('../data/cut_train/train%s.csv' %ix, index=False)
            df_part = pd.DataFrame()


def mkSubFile(lines, head, srcName, sub):
    [des_filename, extname] = os.path.splitext(srcName)
    filename = des_filename + '_' + str(sub) + extname
    print('make file: %s' % filename)
    fout = open(filename, 'w')
    try:
        fout.writelines([head])
        fout.writelines(lines)
        return sub + 1
    finally:
        fout.close()


def splitByLineCount(filename, count):
    fin = open(filename, 'r')
    try:
        head = fin.readline()
        buf = []
        sub = 1
        for line in fin:
            buf.append(line)
            if len(buf) == count:
                sub = mkSubFile(buf, head, filename, sub)
                buf = []
        if len(buf) != 0:
            sub = mkSubFile(buf, head, filename, sub)
    finally:
        fin.close()


if __name__ == '__main__':
    #cut_userfeature
    splitByLineCount('../data/train.csv', 400000)



