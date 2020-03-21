import math, sys
from konlpy.tag import Okt

class BayesianFilter:
    """베이지안 필터"""
    def __init__(self):
        self.words=set() # 출현한 단어 기록
        self.word_dict={} # 카테고리마다의 출현 횟수 기록
        self.category_dict={}
        with open("Stopword.txt",'r',encoding='utf-8') as f: # 불용어 사전 불러와서 리스트화
            self.stopwords=f.read().splitlines()
        
    # 형태소 분석
    def split(self,text):
        results=[]
        okt=Okt()
        malist=okt.pos(text, norm=True)
        for word in malist:
            if word[1] == "Noun": #not in ["Josa","Eomi","Verb","Punctuation"."Adverb"]:
                for s in self.stopwords:
                    judge=''
                    if s in word[0]:
                        judge='Y'
                        break
                if judge != 'Y' :
                    results.append(word[0])
        return results
    
    # 카테고리의 출현 횟수 세기
    def inc_word(self,word,category):
        # 단어를 카테고리에 추가
        if not category in self.word_dict:
            self.word_dict[category]={}
        
        if not word in self.word_dict[category]:
            self.word_dict[category][word]=0
        
        self.word_dict[category][word] += 1
        self.words.add(word)
        
        
    def inc_category(self,category):
        # 카테고리 계산하기
        if not category in self.category_dict:
            self.category_dict[category] = 0
        self.category_dict[category] +=1
        
    # 텍스트 학습하기
    def fit(self,text,category):
        """테스트 학습"""
        word_list=self.split(text)
        for word in word_list:
            self.inc_word(word,category)
        self.inc_category(category)
        
    # 단어 리스트에 점수매기기
    def score(self,words,category):
        score=math.log(self.category_prob(category))
        for word in words:
            score += math.log(self.word_prob(word,category))
        return score
    
    # 예측하기
    def predict(self,text):
        best_category=None
        max_score = -sys.maxsize
        words = self.split(text)
        score_list = []
        for category in self.category_dict.keys():
            score = self.score(words,category)
            score_list.append((category,score))
            if score > max_score:
                max_score = score
                best_category = category
        return best_category, score_list
    
    def get_word_count(self,word,category):
        if word in self.word_dict[category]:
            return self.word_dict[category][word]
        else:
            return 0
        
    # 카테고리 계산 
    def category_prob(self, category):
        sum_categories = sum(self.category_dict.values())
        category_v = self.category_dict[category]
        return category_v / sum_categories
    
    # 카테고리 내부의 단어 출현 비율 계산
    def word_prob(self, word, category):
        n = self.get_word_count(word,category) +1
        d = sum(self.word_dict[category].values()) + len(self.words)
        return n / d