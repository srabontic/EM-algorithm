import sys
import random
#import numpy as np
from math import log, sqrt, exp
import operator

def read_data(fp_data):
    X, Y, train= list(), list(), list()
    for line in fp_data:
        line_arr = line.strip().split(',')
        Y.append(line_arr[0])
        X.append(line_arr[1:16])
        train.append(line_arr)        
    return X, Y, train
    
def init_params(train):
    #initialize parameters
    #print 'init'
    pa = {'1': 0.55, '0': 0.45}
    pb = {'1': {'1': 0.001, '0': 0.999}, '0': {'1': 0.999, '0': 0.001}}
    pc = {'1': {'1': 0001, '0': 0.999}, '0': {'1': 0.1, '0': 0.9}}
    pd = {'1': {'1': 0.001, '2': 0.4, '3': 0.64, '4': 0.5}, '0': {'1': 0.999, '2': 0.6, '3': 0.46, '4': 0.5}}

    
    w_list = list()
    for lines in train:
        w_list.append('1')
    #print w_list
    train1 = list(train)
    counts, count_miss, total = get_data_info(train1, w_list)
    print 'count miss : ' + str(count_miss)
    count_miss1 = count_miss[:]
    count_miss1 = [x / total for x in count_miss1]
    print 'miss prob : ' + str(count_miss1)
    return pa, pb, pc, pd, total 
    #return get_parameters(counts, count_miss, total,train)
    
       
def get_data_info(data, w_list):
    #get total #of datapoints n individual counts
    counts = {}
    total , wt_counter = 0.0, 0
    count_miss = [0]*4
    #count the class
    for lines in data:
        cols = 0
        total +=1
        miss_index = 0
        for colval in lines:
            #print miss_index
            if colval <> '?':
                cols +=1
                counts.setdefault(cols, {})
                counts[cols].setdefault(colval, 0)
                #counts[cols][colval] +=1
                #print w_list[wt_counter]
                counts[cols][colval] += float(w_list[wt_counter])
            elif colval == '?':
                cols +=1
                #get count of missing data
                count_miss[miss_index] +=1
            miss_index +=1
        wt_counter += 1

    #print total
    #print counts
    #print 'count_miss: ' + str(count_miss)
    return counts, count_miss, total
    #get_parameters(counts, count_miss, total,data)

def reevaluate_params(w_list, repl_data, counter_repl_data, total_data_o):
#def get_parameters(counts, count_miss, total,data):
    #print 'w_list : ' + str(w_list)
    count_matrix, count_miss, total_data = get_data_info(repl_data, w_list)
    #print 'new counts: ' + str(count_matrix)
    data = repl_data
    counts = count_matrix
    #print 'count : ' + str(counts)
    total = total_data_o
    #get p(a)
    #print 'counts : ' + str(counts)
    pa = get_pa(data, counts, count_miss, total)
    #print pa
    #get p(b/a)
    i = 1
    j = 0
    pb = get_pb(data, i, j, w_list)
    #get p(c/a)
    i = 2
    j = 0
    pc = get_pb(data, i, j, w_list)
    #get p(d)
    pd = get_pd(data, w_list)
    return pa, pb, pc, pd
    
    
def get_pa(data, counts, count_miss, total):
    #print 'count in pa: ' + str(counts)
    pa = {}
    xa = counts[1]
    #print xa
    #print count_miss
    #print total
    for (attrValue, count) in xa.items():
        pa[attrValue] = count / (total - count_miss[0])
    #print pa
    return pa
       
def get_pb(data,i,j, w_list):
    pb = {}
    c1, c2, c3, c4= 0, 0, 0, 0
    pb.setdefault('1', {})
    pb.setdefault('0', {})
    pb['1'].setdefault('1', {})
    pb['0'].setdefault('0', {})
    w_counter = 0
    for lines in data:
        if lines[i] == '1' and lines[j] == '0':     #b = 1 a =0
            #c1 += 1
            c1 += w_list[w_counter]
            pb['1']['0'] = c1
        elif lines[i] == '1' and lines[j] == '1':     #b = 1 a =1
            #c2 += 1
            c2 += w_list[w_counter]
            pb['1']['1'] = c2
        elif lines[i] == '0' and lines[j] == '1':     #b = 0 a =1
            #c3 +=1
            c3 += w_list[w_counter]            
            pb['0']['1'] = c3
        elif lines[i] == '0' and lines[j] == '0':     #b = 0 a =0
            #c4 += 1
            c4 += w_list[w_counter]            
            pb['0']['0'] = c4
        w_counter += 1
    count_a1 = c1+c2
    count_a2 = c3+c4
    #print 'pb' + str(pb)
    #print 'count_a' + str(count_a)
    for (key, values) in pb.items():
        for (k,pb_counts) in values.items():
            #pb[key][k] = round(pb[key][k] /float(count_a),4)
            print k
            if key == '1':
                pb[key][k] = pb[key][k] /float(count_a1)
            if key =='0':
                pb[key][k] = pb[key][k] /float(count_a2) 
    #print 'cond pb:' + str(pb)
    return pb
    
def get_pd(data, w_list):
    pd = {}
    c01, c02, c03, c04, c11, c12, c13, c14 = 0, 0, 0, 0, 0, 0, 0, 0
    w_counter = 0
    pd.setdefault('1', {})
    pd.setdefault('0', {})
    pd['1'].setdefault('1', {})
    pd['0'].setdefault('1', {})
    for lines in data:
        if lines[3] == '0':     #d
            if lines[1] == '0' and lines[2] == '0':    #b, c
                c01 += w_list[w_counter]
                pd['0']['1'] = c01
            elif lines[1] == '0' and lines[2] == '1': 
                c02 += w_list[w_counter]
                pd['0']['2'] = c02
            elif lines[1] == '1' and lines[2] == '0':
                c03 += w_list[w_counter]
                pd['0']['3'] = c03
            elif lines[1] == '1' and lines[2] == '1':
                c04 += w_list[w_counter]
                pd['0']['4'] = c04
        elif lines[3] == '1':
            if lines[1] == '0' and lines[2] == '0':
                c11 += w_list[w_counter]
                pd['1']['1'] = c11
            elif lines[1] == '0' and lines[2] == '1': 
                c12 += w_list[w_counter]
                pd['1']['2'] = c12
            elif lines[1] == '1' and lines[2] == '0':
                c13 += w_list[w_counter]
                pd['1']['3'] = c13
            elif lines[1] == '1' and lines[2] == '1':
                c14 += w_list[w_counter]
                pd['1']['4'] = c14
        w_counter +=1
    count_d00 = c01+c11
    count_d01 = c02+c12
    count_d10 = c03+c13
    count_d11 = c04+c14
    for (key, values) in pd.items():
        for (k,pd_counts) in values.items():
            #pd[key][k] = round(pd[key][k] /float(count_d),4)
            if k == '1':
                pd[key][k] = pd[key][k] /float(count_d00)
            if k == '2':
                pd[key][k] = pd[key][k] /float(count_d01)
            if k == '3':
                pd[key][k] = pd[key][k] /float(count_d10)
            if k == '4':
                pd[key][k] = pd[key][k] /float(count_d11)
    #print 'pd: ' + str(pd)
    return pd
        
            
def repl_data_weight(data1, total, pa, pb, pc, pd):
    #print 'pa in repl_data-wt: ' + str(pa) + str(pb) + str(pc) + str(pd)
    count =0
    data = list(data1)
    count_repl_data = 0
    miss_in_line = 0
    fixed_wt = 1 
    repl_data = list() 
    weight_list = list()
    count_repl_data = total                   
    for lines in data:
        miss_in_line = lines.count('?')
        #print lines
        #print str(miss_in_line)
        #if lines == '?':
        if miss_in_line > 0:
            #print cols
            if miss_in_line == 1:    # when one data is missing
                temp_list, w_list = get_replaced_value_1(lines)               
                repl_data.extend(temp_list)
                weight_list.extend(w_list)
                count_repl_data += 1 
            elif miss_in_line == 2:  # when two are missing
                temp_list, w_list = get_replaced_value_2(lines)
                repl_data.extend(temp_list)
                weight_list.extend(w_list)
                count_repl_data += 3
            elif miss_in_line == 3:  # when three are missing
                temp_list, w_list = get_replaced_value_3(lines)
                repl_data.extend(temp_list)
                weight_list.extend(w_list)
                count_repl_data += 7                   
        else:
            repl_data.append(lines)         #add one row when data not missing
            weight_list.append(fixed_wt)    #add weight as 1

    return weight_list, repl_data, count_repl_data
                
def get_replaced_value_1(missing_line1):
    missing_line = missing_line1[:]
    temp_list = list()
    weight_list = list()
    #print 'missing_line :' + str(missing_line)
    for i in range(len(missing_line)):
        #print 'i ' + str(i)
        if missing_line[i] == '?':  #when one of the attrs is missing

            #print missing_line[i]
            missing_line[i] = '1'
            m_line1 = missing_line[:]
            #print m_line1
            missing_line[i] = '0'
            m_line2 = missing_line[:]
            #print m_line2
            temp_list.append(m_line1)
            temp_list.append(m_line2)            
    #print temp_list[0]
    for lines in temp_list:    #get w for each new rows
        w = get_row_weights(lines, temp_list, pa, pb, pc, pd)
        weight_list.append(w)
    #print m_line1, m_line2
    return temp_list, weight_list

    
def get_replaced_value_2(missing_line1):
    missing_line = missing_line1[:]
    temp_list, temp_temp_list = list(), list()
    weight_list = list()
    miss_index1, miss_index2 = 0, 0
    #replace missing data
    for i in range(len(missing_line)):
        if missing_line[i] == '?':  #when one of the attrs is missing
            #miss_index1 = i
            missing_line[i] = '1'
            m_line1 = missing_line[:]
            missing_line[i] = '0'
            m_line2 = missing_line[:]
            break
    temp_list.append(m_line1)
    temp_list.append(m_line2)

    for lines in temp_list:
        for i in range(len(lines)):
            if lines[i] == '?':  #when one of the attrs is missing
                lines[i] = '1'
                m_line3 = lines[:]
                lines[i] = '0'
                m_line4 = lines[:]
        temp_temp_list.append(m_line3)
        temp_temp_list.append(m_line4)
         
    for lines in temp_temp_list:    #get w for each new rows
        w = get_row_weights(lines, temp_temp_list, pa, pb, pc, pd)
        weight_list.append(w)
    #print '2 missing: ' + str(temp_temp_list)
    return temp_temp_list, weight_list
    
def get_replaced_value_3(missing_line1):
    missing_line = missing_line1[:]
    temp_list, temp_list1, temp_list2 = list(), list(), list()
    weight_list = list()
    #miss_index1, miss_index2 = 0, 0
    #replace missing data
    for i in range(len(missing_line)):
        if missing_line[i] == '?':  #when one of the attrs is missing
            #miss_index1 = i
            missing_line[i] = '1'
            m_line = missing_line[:]
            temp_list.append(m_line)
            missing_line[i] = '0'
            m_line = missing_line[:]            
            temp_list.append(m_line)
            break
            
    for lines in temp_list:
        set_flag = 0
        for i in range(len(lines)):
            if lines[i] == '?' and set_flag == 0:  #when one of the attrs is missing
                #miss_index2 = i
                lines[i] = '1'
                m_line = lines[:]
                temp_list1.append(m_line)
                lines[i] = '0'
                m_line = lines[:]
                temp_list1.append(m_line)
                set_flag = 1

    for lines in temp_list1:
        set_flag = 0
        for i in range(len(lines)):
            if lines[i] == '?' and set_flag == 0:  #when one of the attrs is missing
                lines[i] = '1'
                m_line = lines[:]
                temp_list2.append(m_line)
                lines[i] = '0'
                m_line = lines[:]
                temp_list2.append(m_line)
                set_flag = 1

    for lines in temp_list2:    #get w for each new rows
        w = get_row_weights(lines, temp_list2, pa, pb, pc, pd)
        weight_list.append(w)
    #print '3 missing' + str(temp_list2)
    return temp_list2, weight_list
    
def get_row_weights(temp_list_row,temp_list, pa, pb, pc, pd):  #receives one row of temp_list
    w, wn, wd = 0, 0, 0
    a = str(temp_list_row[0])
    b = str(temp_list_row[1])
    c = str(temp_list_row[2])
    d = str(temp_list_row[3])
    bc = str(get_bc(b,c))
    wn = (pa[a]*pb[b][a]*pc[c][a]*pd[d][bc])
    for rows in temp_list:
        a = str(rows[0])
        b = str(rows[1])
        c = str(rows[2])
        d = str(rows[3])
        bc = str(get_bc(b,c))
        wd += pa[a]*pb[b][a]*pc[c][a]*pd[d][bc]
    w = round((wn/ wd), 2)    # calculate w for one row of temp_list
    return w
     

    
def calc_loglikelihood(pa, pb, pc, pd, data):
    l = 0.0
    for lines in data:
        a = str(lines[0])
        b = str(lines[1])
        c = str(lines[2])
        d = str(lines[3])
        bc = str(get_bc(b,c))
        #if pa[a]*pb[b][a]*pc[c][a]*pd[d][bc] <> 0:
        #   l += log((pa[a]*pb[b][a]*pc[c][a]*pd[d][bc]))
        if pa[a]<> 0:
            l+= log(pa[a])
        if pb[b][a] <> 0:
            l += log(pb[b][a])
        if pc[c][a] <> 0:
            l += log(pc[c][a])
        if pd[d][bc] <> 0:
            l += log(pd[d][bc])
    return l
    
def get_bc(b,c):
    if b == '0' and c == '0':
        bc = 1
    elif b == '0' and c == '1': 
        bc = 2
    elif b == '1' and c == '0': 
        bc = 3
    elif b == '1' and c == '1': 
        bc = 4
    return bc
                    
if __name__ == '__main__':    
    train_path = 'C:\Users\User\Desktop\MS everything\MC learning\HW\PS5\Ebn.txt';
    X_train, Y_train, train = read_data(open(train_path))
    e, l, l0= 0.001, -500.0, 0.0
    #get theta0
    pa, pb, pc, pd, total_count_data =init_params(train)
    #l = calc_loglikelihood(pa, pb, pc, pd, train)
    total_ori_ct = total_count_data
    set_1 = 1
    #E step to complete data set based on theta
    for i in range(1,20):
        #print i
        if abs(l - l0) > e or set_1 == 1:
            #pa, pb, pc, pd = pai, pbi, pci, pdi
            train_data = list(train)
            wt_list, repl_data, repl_data_count = repl_data_weight(train_data, total_count_data, pa, pb, pc, pd)
            #print 'after estimate w: ' + str(wt_list)
            #M step to re-estimate thetas'
            pa, pb, pc, pd = reevaluate_params(wt_list, repl_data, repl_data_count, total_ori_ct)
            print 'P(A) : ' + str(pa)
            print 'P(B|A) : ' + str(pb)
            print 'P(C|A) : ' + str(pc)
            print 'P(D|B,C) : ' + str(pd)
            l0 = l
            repl_data1 = repl_data[:]
            l = calc_loglikelihood(pa, pb, pc, pd,repl_data1) 
            if set_1 == 1:
                set_1 = 0 
            print 'log lokelihood in itr'+str(i)+ ':  ' + str(l)
            #print 'l0: ' + str(l0)
               