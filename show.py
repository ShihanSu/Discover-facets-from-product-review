
def txt2dict(file):
    ''' Parse segment file string into dictionary'''
    f = open(file)
    content = f.read().split()
    data = {}
    keys = ''
    topics = ['feel', 'taste','look','overall','smell']
    
    for i in content:
        if i in topics:
            keys = i
            data[keys] = []
        else:
            data[keys].append(float(i))
    return data
    

def txt2dict_sen(file):
    ''' Parse segment file string into dictionary, single dimension rating'''
    f = open(file)
    content = f.read().split()
    data = {}
    keys = ''
    topics = ['feel', 'taste','look','overall','smell']
    ratings  = ['0', '1']
    
    for i in content:
        if i in ratings:
            keys = i
            data[keys] = []
        elif i not in topics:
            data[keys].append(float(i))
    return data
    

def txt2dict_sen2(file):
    ''' Parse sentiment file string into dictionary, rating with hidden dimension'''
    f = open(file)
    content = f.read().split()
    data = {}
    keys = ''
    
    subkeys = ''
    topics = ['feel', 'taste','look','overall','smell']
    ratings  = ['0', '1']
    
    
    for i in content:
        if i in topics:
            keys = i
            data[keys] = {}
            
            for i in content:
                num = 0
                if i in ratings:
                    subkeys = i
                    data[keys][subkeys] = []
                    num += 1

                elif i not in topics and num <= 6:
                    data[keys][subkeys].append(float(i))
            
    return data
    
# extract ids
f = open('/wordids/wordidsBA.txt')
ids = []
for line in f.readlines():
        ids.append(line.split()[1])
        
def dict2df(dicts, ids):
    import pandas as pd
    data = pd.DataFrame.from_dict(dicts)
    data['voc'] = ids
    data['voc'] = data['voc'].str.replace('"', '')
    
    return data
    
def createFile(attr,target, rating = 1):
    if target == 'topic': 
        data = txt2dict('output/modelSegBA.out')
        datadf = dict2df(data, ids)
        
        df = datadf[['voc', attr]]
        df.to_csv('%s_%s.txt'%(target, attr),header = None, index = None, sep = ':')
        
    if target == 'sentiment':
        data = txt2dict_sen2('output/modelSenBA.out')
        datadf = dict2df(data[attr], ids)
        
        df = datadf[['voc', rating]]
					
        df.loc[:, rating] = df[rating] + 0.5
        df.to_csv('%s_%s.txt'%(attr, rating),header = None, index = None, sep = ':')
        


topics = ['feel', 'taste','look','overall','smell']
ratings  = ['0', '1']
types = ['topic', 'sentiment']
    
for i in topics:
	for j in types:
		if j == 'topic':
			createFile(i, j)
		else:
			for k in ratings:
				createFile(i, j, k)
				
