import numpy as np
import os
import pickle
import linecache
import math
import lightgbm as lgb
import joblib
import argparse

class DNAlight(object):
    def __init__(self,fastapath, pdbid, psiblastoutpath, PSSMpath, psipredoutpath):
        '''
        fastapath: where you put your fasta file
        pdbid: protein id of fasta file
        psiblastoutpath: path for output file of PSI-Blast
        PSSMpath: path for the PSSM file
        psipredoutpath: path for output file of psipred'
        '''        
        self.fastapath = fastapath
        self.pdbid = pdbid
        self.psiblastoutpath = psiblastoutpath
        self.PSSMpath = PSSMpath
        self.psipredoutpath = psipredoutpath
        
    def PSSMfeature(self):
        cmd='/home/songjiazhi/blast/bin/psiblast -evalue 10 -num_iterations 3 -db /home/songjiazhi/blast/db/uniprot -query '+self.fastapath+'/'+self.pdbid+'.fasta'+' -outfmt 0 -out '+self.psiblastoutpath+'/'+self.pdbid+'.fm0 -out_ascii_pssm '+self.PSSMpath+'/'+self.pdbid+'.pssm -num_alignments 1500 -num_threads 8'
        '''
        change to your own PSI-Blast dir
        the referenced database is availabel at $HOME/tool/blast/db/uniprot
        '''
        os.system(cmd)  
        pssmfilelines = linecache.getlines(self.PSSMpath+'/'+self.pdbid+'.pssm')
        pssmDic = {}
        for line in pssmfilelines:
            content = line.split()
            if len(content) == 44:
                residuePosition = int(content[0])-1
                pssmDic[residuePosition] = []
                for i in range(2,22):
                    #pssmDic[str(residuePosition)].append(int(content[i]))
                    pssmDic[residuePosition].append(self.normalize(int(content[i])))     
        return pssmDic
    
    def normalize(self,value):
        a = 1+math.exp(value)
        b = 1/a
        return b        
    
    def psipredfeature(self):
        cmd = '/home/songjiazhi/psipred.4.02/psipred/BLAST+/runpsipredplus '+self.fastapath+'/'+self.pdbid+'.fasta'
        '''change to your psipred dir'''
        os.system(cmd)        
        psipredDic = {}
        filelines = linecache.getlines(self.psipredoutpath+'/'+self.pdbid+'.ss2')
        length = len(filelines)
        for i in range(2,length):
            residuenum = int(filelines[i].split()[0])-1
            psipredDic[residuenum] = []
            psipredDic[residuenum].append(float(filelines[i].split()[3]))
            psipredDic[residuenum].append(float(filelines[i].split()[4]))
            psipredDic[residuenum].append(float(filelines[i].split()[5]))
        return psipredDic   
    
    def onehotfeature(self):
        fastalines = linecache.getlines(self.fastapath+'/'+self.pdbid+'.fasta')
        chemicaldic = {}
        fastaline = fastalines[1]
        if fastaline[-1] == '\n':
            fastaline = fastaline[:-1]
        length = len(fastaline)
        for i in range (length):
            residuename = fastaline[i]
            if residuename == 'A' or residuename == 'G' or residuename == 'V':
                chemicaldic[i] = [0,0,0,0,0,0,1]
            elif residuename == 'I' or residuename == 'L' or residuename == 'F' or residuename == 'P':
                chemicaldic[i] = [0,0,0,0,0,1,0]
            elif residuename == 'H' or residuename == 'N' or residuename == 'Q' or residuename == 'W':
                chemicaldic[i] = [0,0,0,0,1,0,0]
            elif residuename == 'Y' or residuename == 'M' or residuename == 'T' or residuename == 'S':
                chemicaldic[i] = [0,0,0,1,0,0,0]
            elif residuename == 'R' or residuename == 'K':
                chemicaldic[i] = [0,0,1,0,0,0,0]
            elif residuename == 'D' or residuename == 'E':
                chemicaldic[i] = [0,1,0,0,0,0,0]
            elif residuename == 'C':
                chemicaldic[i] = [1,0,0,0,0,0,0]
            elif residuename == 'U':
                chemicaldic[i] = [0,0,0,0,0,0,0] 
        return chemicaldic
    
    def physiochemicalfeature(self):
        pkadic = {'G':[2.34,9.60,7.00],
                  'A':[2.34,9.69,7.00],
                  'P':[1.99,10.96,7.00],
                  'V':[2.32,9.62,7.00],
                  'L':[2.36,9.60,7.00],
                  'I':[2.36,9.68,7.00],
                  'M':[2.28,9.21,7.00],
                  'F':[1.83,9.13,7.00],
                  'Y':[2.20,9.11,10.07],
                  'W':[2.38,9.39,7.00],
                  'S':[2.21,9.15,7.00],
                  'T':[2.11,9.62,7.00],
                  'C':[1.96,10.28,8.18],
                  'N':[2.02,8.80,7.00],
                  'Q':[2.17,9.13,7.00],
                  'K':[2.18,8.95,10.53],
                  'H':[1.82,9.17,6.00],
                  'R':[2.17,9.04,12.48],
                  'D':[1.88,9.60,3.65],
                  'E':[2.19,9.67,4.25]}
        mmassdic = {'G':75,
                    'A':89,
                    'P':115,
                    'V':117,
                    'L':131,
                    'I':131,
                    'M':149,
                    'F':165,
                    'Y':181,
                    'W':204,
                    'S':105,
                    'T':119,
                    'C':121,
                    'N':132,
                    'Q':146,
                    'K':146,
                    'H':155,
                    'R':174,
                    'D':133,
                    'E':147}
        EIIPdic = {'G':0.0050,
                   'A':0.0373,
                   'P':0.0198,
                   'V':0.0057,
                   'L':0.0000,
                   'I':0.0000,
                   'M':0.0823,
                   'F':0.0946,
                   'Y':0.0516,
                   'W':0.0548,
                   'S':0.0829,
                   'T':0.0941,
                   'C':0.0829,
                   'N':0.0036,
                   'Q':0.0761,
                   'K':0.0371,
                   'H':0.0242,
                   'R':0.0959,
                   'D':0.1263,
                   'E':0.0058}
        #chemicaldic = [hydrophobicity,hydrophilicity,polarity,polarizability,average accessible surface area]
        aaindexdic = {'G':[0.07,0.0,9.0,0.000,24.5],
                       'A':[0.61,-0.5,8.1,0.046,27.8],
                       'P':[1.95,0.0,8.0,0.131,51.5],
                       'V':[1.32,-1.5,5.9,0.140,23.7],
                       'L':[1.53,-1.8,4.9,0.186,27.6],
                       'I':[2.22,-1.8,5.2,0.186,22.8],
                       'M':[1.18,-1.3,5.7,0.221,33.5],
                       'F':[2.02,-2.5,5.2,0.290,25.5],
                       'Y':[1.88,-2.3,6.2,0.298,55.2],
                       'W':[2.65,-3.4,5.4,0.409,34.7],
                       'S':[0.05,0.3,9.2,0.062,42.0],
                       'T':[0.05,-0.4,8.6,0.108,45.0],
                       'C':[1.07,-1.0,5.5,0.128,15.5],
                       'N':[0.06,0.2,11.6,0.134,60.1],
                       'Q':[0.00,0.2,10.5,0.180,68.7],
                       'K':[1.15,3.0,11.3,0.219,103.0],
                       'H':[0.61,-0.5,10.4,0.230,50.7],
                       'R':[0.60,3.0,10.5,0.291,94.7],
                       'D':[0.46,3.0,13.0,0.105,60.6],
                       'E':[0.47,3.0,12.3,0.151,68.2]}        
        fastalines = linecache.getlines(self.fastapath+'/'+self.pdbid+'.fasta')
        physiochemicaldic = {}
        fastaline = fastalines[1]
        if fastaline[-1] == '\n':
            fastaline = fastaline[:-1]
        length = len(fastaline) 
        for i in range(length):
            residuename = fastaline[i]
            physiochemicaldic[i] = []
            for a in range(len(pkadic[residuename])):
                physiochemicaldic[i].append(pkadic[residuename][a])
            physiochemicaldic[i].append(mmassdic[residuename])
            physiochemicaldic[i].append(EIIPdic[residuename])
            for b in range(len(aaindexdic[residuename])):
                physiochemicaldic[i].append(aaindexdic[residuename][b]) 
        return physiochemicaldic
    
    def featurecombine(self):
        pssmdic = self.PSSMfeature()
        psipreddic = self.psipredfeature()
        chemicaldic = self.onehotfeature()
        physiochemicaldic = self.physiochemicalfeature()
        length = len(pssmdic.keys())
        featuredic = {}
        for i in range(length):
            featuredic[i] = []
            for each in pssmdic[i]:
                featuredic[i].append(each)
            for each in psipreddic[i]:
                featuredic[i].append(each)
            for each in chemicaldic[i]:
                featuredic[i].append(each) 
            for each in physiochemicaldic[i]:
                featuredic[i].append(each)
        appendedfeaturedic = self.appendzero(17,featuredic)
        combinefeaturedic = self.combine(length, appendedfeaturedic, 17)
        return combinefeaturedic 
    
    def appendzero(self,windowsize,featureDic):
        seqlength = len(featureDic.keys())
        appendnum = int((windowsize+1)/2)
        for i in range(1,appendnum):
            featureDic[0-i] = []
            featureDic[seqlength-1+i] = []
            for a in range(40):
                featureDic[0-i].append(0)
            for b in range(40):
                featureDic[seqlength-1+i].append(0)
        return featureDic    

    def combine(self,sequencelength,featuredic,windowsize):
        neighnum = int((windowsize-1)/2)
        combineDic = {}
        for i in range(0,sequencelength):
            combineDic[i] = []
            for a in range(i - neighnum,i + neighnum + 1):
                #combineDic[i].append(pssmdic[a])
                for each in featuredic[a]:
                    combineDic[i].append(each)
        featurelist = []
        for i in range(0,sequencelength):
            featurelist.append(combineDic[i])
        return featurelist
    
    def lgbprediction(self,featuredic):
        lgbmodelpickle = open('/home/newdisk/songjiazhi/DNAdeep/paper/lgb_model.m','rb')
        lgbmodel = pickle.load(lgbmodelpickle)
        lgbprediction = lgbmodel.predict(featuredic, num_iteration=500)
        return lgbprediction
    
    def stage1(self, lgbprediction, lgbthreshold):
        fastalines = linecache.getlines(self.fastapath+'/'+self.pdbid+'.fasta')
        fastaline = fastalines[1]
        if fastaline[-1] == '\n':
            fastaline = fastaline[:-1] 
        residuebindingprobdic_pickle = open('/home/newdisk/songjiazhi/DNAdeep/paper/bindingprobdic.pickle','rb')
        residuebindingprobdic = pickle.load(residuebindingprobdic_pickle)
        stage1prediction = []
        for i in range(len(fastaline)):
            if lgbprediction[i] < lgbthreshold:    
                if i == 0:
                    residuename = fastaline[i]
                    rightresiduename = fastaline[i+1]
                    rightprob = residuebindingprobdic[residuename][rightresiduename]
                    if rightprob > 0.17:
                        stage1prediction.append(lgbprediction[i]+rightprob*lgbprediction[i+1])
                    else:
                        stage1prediction.append(lgbprediction[i])
                elif i == len(fastaline)-1:
                    residuename = fastaline[i]
                    leftresiduename = fastaline[i-1]
                    leftprob = residuebindingprobdic[residuename][leftresiduename]
                    if leftprob > 0.17:
                        stage1prediction.append(lgbprediction[i]+leftprob*lgbprediction[i-1])
                    else:
                        stage1prediction.append(lgbprediction[i])
                else:
                    residuename = fastaline[i]
                    leftresiduename = fastaline[i-1]
                    leftprob = residuebindingprobdic[residuename][leftresiduename]
                    rightresiduename = fastaline[i+1]
                    rightprob = residuebindingprobdic[residuename][rightresiduename]
                    if leftprob >= rightprob:
                        if leftprob > 0.17:
                            stage1prediction.append(lgbprediction[i]+leftprob*lgbprediction[i-1])
                        else:
                            stage1prediction.append(lgbprediction[i])
                    else:
                        if rightprob > 0.17:
                            stage1prediction.append(lgbprediction[i]+rightprob*lgbprediction[i+1])
                        else:
                            stage1prediction.append(lgbprediction[i])
            else:
                stage1prediction.append(lgbprediction[i])
        return stage1prediction
    
    def stage2(self, stage1prediction, lgbthreshold):
        fastalines = linecache.getlines(self.fastapath+'/'+self.pdbid+'.fasta')
        fastaline = fastalines[1]
        if fastaline[-1] == '\n':
            fastaline = fastaline[:-1] 
        residuebindingprobdic_pickle = open('/home/newdisk/songjiazhi/DNAdeep/paper/bindingprobdic_2.pickle','rb')
        residuebindingprobdic = pickle.load(residuebindingprobdic_pickle)
        stage2prediction = []      
        for i in range(len(fastaline)):
            if stage1prediction[i] < lgbthreshold:
                if i == 0 or i == 1:
                    residuename = fastaline[i]
                    rightresiduename = fastaline[i+1]
                    rightprob = residuebindingprobdic[residuename][rightresiduename]
                    if rightprob > 0.20:
                        stage2prediction.append(stage1prediction[i]+rightprob*stage1prediction[i+1])
                    else:
                        stage2prediction.append(stage1prediction[i])
                elif i == len(fastaline)-1:
                    residuename = fastaline[i]
                    leftresiduename = fastaline[i-1]
                    leftprob = residuebindingprobdic[residuename][leftresiduename]
                    if leftprob > 0.20:
                        stage2prediction.append(stage1prediction[i]+leftprob*stage1prediction[i-1])
                    else:
                        stage2prediction.append(stage1prediction[i])
                else:
                    residuename = fastaline[i]
                    leftresiduename = fastaline[i-1]
                    leftprob = residuebindingprobdic[residuename][leftresiduename]
                    rightresiduename = fastaline[i+1]
                    rightprob = residuebindingprobdic[residuename][rightresiduename]
                    if leftprob >= rightprob:
                        if leftprob > 0.20:
                            stage2prediction.append(stage1prediction[i]+leftprob*stage1prediction[i-1])
                        else:
                            stage2prediction.append(stage1prediction[i])
                    else:
                        if rightprob > 0.20:
                            stage2prediction.append(stage1prediction[i]+rightprob*stage1prediction[i+1])
                        else:
                            stage2prediction.append(stage1prediction[i])
            else:
                stage2prediction.append(stage1prediction[i])
        return stage1prediction        
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fasta', help='fasta file path')
    parser.add_argument('-i', '--pdbid', help='fasta file id')
    args = parser.parse_args()
    
    fastapath = args.fasta
    pdbid = args.pdbid
    predict = DNAlight(fastapath, pdbid, '/home/newdisk/songjiazhi/DNAdeep/paper/blastout/out', '/home/newdisk/songjiazhi/DNAdeep/paper/blastout/pssm', '/home/newdisk/songjiazhi/DNAdeep/paper')
    '''change dirs according to the definations in DNAlight'''
    featurelist = predict.featurecombine()
    #featurelistpickle = open('/home/newdisk/songjiazhi/DNAdeep/paper/trainfeatureexample.pickle','wb')
    #pickle.dump(featurelist, featurelistpickle)
    lgbprediction = predict.lgbprediction(featurelist)
    print(len(featurelist[0]))
    stage1prediction = predict.stage1(lgbprediction,0.48)
    stage2prediction = predict.stage2(stage1prediction,0.48)
    length = len(stage2prediction)
    predictfile = open('/home/newdisk/songjiazhi/DNAdeep/paper/'+pdbid+'_prediction.txt','w')
    predictfile.write('residue'+'  '+'prediction')
    predictfile.write('\n')
    for i in range(length):
        predictfile.write(str(i)+' '+str(stage2prediction[i]))
        predictfile.write('\n') 
    predictfile.close()
    
if __name__ == "__main__":
    main()
            
                    
        
        