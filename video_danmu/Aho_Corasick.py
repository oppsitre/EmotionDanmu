#Python3实现AC自动机
#http://www.ctolib.com/topics-106266.html

#结点类
class node:
    def __init__(self,ch):
        self.ch = ch            #结点值
        self.fail = None        #Fail指针
        self.tail = 0           #尾标志：标志为 i 表示第 i 个模式串串尾
        self.child = []         #子结点
        self.childvalue = []    #子结点的值

#AC自动机类
class acmation:
    def __init__(self):
        self.root = node("")                      #初始化根结点
        self.count = 0                            #模式串个数

    #第一步：模式串建树
    def insert(self,strkey):
        self.count += 1                             #插入模式串，模式串数量加一
        p = self.root
        for i in strkey:
            if i not in p.childvalue:               #若字符不存在，添加子结点
                child = node(i)
                p.child.append(child)
                p.childvalue.append(i)
                p = child
            else :                                  #否则，转到子结点  #point to child node
                p = p.child[p.childvalue.index(i)]
        p.tail = strkey                         #修改尾标志

    #第二步：修改Fail指针
    def ac_automation(self):
        queuelist = [self.root]                     #用列表代替队列
        while len(queuelist):                       #BFS遍历字典树
            temp = queuelist[0]
            queuelist.remove(temp)                  #取出队首元素
            for i in temp.child:
                if temp == self.root:               #根的子结点Fail指向根自己
                    i.fail = self.root
                else:
                    p = temp.fail                   #转到Fail指针
                    while p:
                        if i.ch in p.childvalue:    #若结点值在该结点的子结点中，则将Fail指向该结点的对应子结点
                            i.fail = p.child[p.childvalue.index(i.ch)]
                            break
                        p = p.fail                  #否则，转到Fail指针继续回溯
                    if not p:                       #若p==None，表示当前结点值在之前都没出现过，则其Fail指向根结点
                        i.fail = self.root
                queuelist.append(i)                 #将当前结点的所有子结点加到队列中

    #第三步：模式匹配
    def runkmp(self,strmode):
        p = self.root
        cnt = []                                   #使用字典记录成功匹配的状态
        for j in range(len(strmode)):              #遍历目标串
            i = strmode[j]
            while i not in p.childvalue and p is not self.root:
                p = p.fail
            if i in p.childvalue:                   #若找到匹配成功的字符结点，则指向那个结点，否则指向根结点
                p = p.child[p.childvalue.index(i)]
            else :
                p = self.root
            temp = p

            while temp is not self.root:
                if temp.tail:                      #尾标志为0不处理
                    cnt.append((temp.tail, j-len(temp.tail)+1, j))
                temp = temp.fail

        return cnt                                   #返回匹配状态
        #如果只需要知道是否匹配成功，则return bool(cnt)即可
        #如果需要知道成功匹配的模式串种数，则return len(cnt)即可

    def expression_extract(self, strmode):
        #keep [[殷俊俊]] rather than [[殷俊],[殷俊俊]]
        express = self.runkmp(strmode)
        dedup_res = []

        if express == []: return [(strmode, 'str')]

        express = sorted(express, key=lambda x:x[1])
        temp_start, temp_end = 0, 0; temp_exp = None
        for i in express:
            if i[1]==temp_start and i[2] >= temp_end:
                temp_end = i[2]
                temp_exp = i
            if i[1] > temp_end:
                if temp_exp!=None: dedup_res.append(temp_exp)
                temp_start,temp_end = i[1],i[2]
                temp_exp = i
        dedup_res.append(temp_exp)
        # print(dedup_res)
        # print('dedup_res', len(dedup_res))

        start = [i[1] for i in dedup_res]
        end = [i[2] for i in dedup_res]
        strings = []

        for i in range(0,len(start)):
            if i==0 :
                string = strmode[:start[i]]
            else:
                string = strmode[end[i-1]+1:start[i]]
            expres = strmode[start[i]:end[i]+1]
            if string!='': strings.append((string, 'str'))
            if expres!='': strings.append((expres, 'exp'))

        if  end[-1]+1 <= len(strmode):
        	string = strmode[end[i]+1:]
        	strings.append((string, 'str'))

        return strings

#同時出現兩個expression甘點分? eg. (╯°口°)╯(╯°口°)╯qw殷俊
#same thing happens in BagOfWords!!

'''
key = ["殷俊","王志青","dahai","qww","殷俊俊","(╯°口°)╯","(ﾟДﾟ≡ﾟдﾟ)","°口°)","(╯°口°)╯(╯°口°)╯", "(╯°口°)╯qw"]        #创建模式串
acp = acmation()

for i in key:
    acp.insert(i)                           #添加模式串

acp.ac_automation()

text = '(╯°口°)╯(╯°口°)╯qw殷俊俊殷俊俊qqqwweq学习资源、王志青等wj 的置顶功能dahai,殷俊dahabekgwbnudaihai王志清殷殷俊(ﾟДﾟ≡ﾟдﾟ)'
d = acp.runkmp(text)                        #运行自动机
e = acp.expression_extract(text)

'''
