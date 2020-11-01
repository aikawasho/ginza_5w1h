import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token
import re
import pytextrank

class phrase_5w1h(object):
    def __init__(self,phrase = None, start = None, end = None,_type = None):
        self.phrase = phrase
        self.start = start
        self.end = end
        self._type = _type
        
#
class parse_5w1h(object):

    def __init__(self,doc = None):
        self.doc = doc
        

    def extract(self,text):
        #構文解析器のインスタンス化
        nlp = spacy.load('ja_ginza')
        #マッチャーインスタンス化
        matcher = Matcher(nlp.vocab)
        #TextRank
        tr = pytextrank.TextRank()
        nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)
        
        if not Token.has_extension("_5w1h"):
            Token.set_extension("_5w1h", default='None')
            
        if not Token.has_extension("_type"):
            Token.set_extension("_type", default=None)


         #5w1hパターン
        Where_pattern1 = [{"POS":{"REGEX":"NOUN|PROPN|PRON"},"ENT_TYPE":{"REGEX":"Country|Province"}}]
        Where_pattern2 = [{"TEXT":{"REGEX":"Amazon"}}]
        Where_pattern3 = [{"TEXT":"中"},{"TEXT":"に"}]

        When_pattern1 = [{"TEXT":{"REGEX":"今日|昨日|おととい|明日|先日|明後日|今回|前回|後ほど|その後"}}]
        When_pattern2 = [{"POS":"NUM"},{"TEXT":{"REGEX":"時|日|月|年"},"LENGTH":1}]
        When_pattern3 = [{"TEXT":{"REGEX":"今"},"LENGTH":1}]
        When_pattern4 = [{"TEXT":"この"},{"TEXT":{"REGEX":"後|前"}}] 

        Why_pattern1 = [{"POS":{"REGEX":"VERB|ADJ"},"OP":"+"},{"POS":{"REGEX":"AUX"},"OP":"?"},{"TEXT":"から"}]


        How_pattern1 = [{"POS":{"REGEX":"VERB|ADJ|AUX|PART"},"DEP":{"REGEX":"ROOT|punct"}}]
        How_pattern2 = [{"POS":{"REGEX":"VERB|ADJ"},"OP":"+"},{"DEP":{"REGEX":"ROOT|cc"}}]
        How_pattern3 = [{"POS":{"REGEX":"VERB|ADJ|NOUN"},"OP":"+"},{"POS":{"REGEX":"AUX|ADP"},"OP":"?"},{"DEP":{"REGEX":"ROOT"}}]
        How_pattern4 = [{"POS":{"REGEX":"VERB|ADJ"},"OP":"+"},{"OP":"?"},{"POS":"PUNCT"}]
        How_pattern5 = [{"POS":{"REGEX":"VERB|ADJ|AUX"},"OP":"*"},{"TEXT":"の"},{"TEXT":"に"}]
        How_pattern6 = [{"POS":"VERB","OP":"?"},{"POS":"AUX","OP":"+"},{"TEXT":"が","POS":"CCONJ"},{"POS":"PUNCT","OP":"?"}]
        How_pattern7 = [{"POS":"VERB"},{"POS":"AUX"},{"TEXT":"ところ","POS":"NOUN"}]
        How_pattern8 = [{"POS":"VERB","DEP":"advcl"},{"TEXT":"し","POS":"CCONJ"}]
        How_pattern9 = [{"POS":{"REGEX":"VERB|ADJ"}},{"TEXT":"ん","OP":"?"},{"TEXT":"です","OP":"?"},{"LEMMA":"けれど"},{"TEXT":"も","OP":"?"},{"TEXT":"、","OP":"?"}]
        How_pattern10 = [{"POS":"AUX","TEXT":{"NOT_IN":["で"]}},{"POS":{"REGEX":"PUNCT"}}]
        How_pattern11 = [{"POS":{"REGEX":"VERB|ADJ|AUX"},"OP":"*"},{"TEXT":"けれど"},{"TEXT":"も"}]
        How_pattern12 = [{"POS":"VERB","OP":"?"},{"POS":"AUX","OP":"+"},{"TEXT":"と"},{"TEXT":"か","OP":"?"},{"POS":"PUNCT","OP":"?"}]
        How_pattern13 = [{"POS":"AUX"},{"TEXT":"が","POS":{"REGEX":"CCONJ"}}]
        How_pattern14 = [{"POS":"NOUN"},{"TEXT":"です"},{"POS":{"REGEX":"PART"},"OP":"*"}]
        How_pattern15 = [{"POS":"NOUN"},{"TEXT":"か"},{"TEXT":"な"},{"POS":{"REGEX":"PUNCT"}}]

        Who_pattern1 = [{"TEXT":"の","OP":"?"},{"POS":"NOUN","DEP":"compound","OP":"*"},{"POS":{"REGEX":"NOUN|PRON|PROPN"},"DEP":{"REGEX":"iobj|obl|nsubj"},"TAG":{"NOT_IN":["名詞-普通名詞-助数詞可能"]},"TEXT":{"NOT_IN":["幾つ"]}},{"TEXT":{"REGEX":"が|は|も"}},{"TEXT":{"REGEX":"です|ね|、"},"OP":"*"}]
        Who_pattern2 = [{"DEP":{"REGEX":"amod|advmod|acl"},"OP":"+"},{"TEXT":"ところ","DEP":{"REGEX":"compound"}}]
        Who_pattern3 = [{"POS":{"REGEX":"NOUN|PRON|PROPN"},"DEP":{"REGEX":"iobj|obl|nsubj"}},{"TEXT":{"REGEX":"に"}},{"TEXT":{"REGEX":"は"}}]
        Who_pattern4 = [{"POS":{"REGEX":"NOUN|PRON|PROPN"},"DEP":{"REGEX":"obl|nmod|dep"},"TEXT":{"NOT_IN":["幾つ"]},"TAG":{"NOT_IN":["名詞-普通名詞-助数詞可能"]}},{"TEXT":{"REGEX":"が|は|も|って"}}]
        Who_pattern5 = [{"TEXT":"こと","DEP":"compound"},{"TEXT":{"REGEX":"が|は|も"}}]
        Who_pattern6 = [{"POS":"NUM","OP":"!"},{"POS":{"REGEX":"NOUN|PRON|PROPN"},"DEP":{"REGEX":"iobj|obl|nsubj"},"TAG":"名詞-普通名詞-助数詞可能"},{"TEXT":{"REGEX":"が|は|も"}}]
        Who_pattern7 = [{"POS":{"REGEX":"VERB|ADJ"},"OP":"+"},{"POS":"NOUN","OP":"?"},{"TEXT":"の"},{"TEXT":"が"}]
        Who_pattern8 = [{"TEXT":"と"},{"TEXT":"いう"},{"TEXT":"の"},{"TEXT":"は"}]
        Who_pattern9 =[{"POS":"NOUN"},{"TEXT":"に"},{"TEXT":"おい"},{"TEXT":"て"},{"TEXT":"は"}]

        What_pattern1 = [{"POS":{"REGEX":"NOUN|PRON|PROPN"},"DEP":{"REGEX":"obl|obj|iobj"}},{"TEXT":{"REGEX":"を"}}]
        What_pattern2 = [{"POS":{"REGEX":"SYM"},"DEP":{"REGEX":"dep"}},{"TEXT":{"REGEX":"を"}}]


        Mod_pattern1 = [{"DEP":{"REGEX":"amod|advmod|nmod|case|obl|case|acl|aux|det|nsubj|dep|mark|compound|nummod|advcl|iobj|det|obj"}}]
        Mod_pattern2 = [{"TEXT":"いつ"}]
        
        Task_pattern1 = [{"TEXT":"たい"},{"TEXT":"の"},{"POS":{"REGEX":"PUNCT"},"OP":"*"}]
        Task_pattern2 = [{"TEXT":"ない"},{"TEXT":"か"},{"TEXT":"な"},{"POS":{"REGEX":"PUNCT"},"OP":"*"}]
        Task_pattern3 = [{"POS":{"REGEX":"VERB|AUX"}},{"TEXT":"ない"},{"TEXT":"で"},{"TEXT":"ね"},{"POS":{"REGEX":"PUNCT"},"OP":"*"}]
        Task_pattern4 = [{"POS":"AUX"},{"POS":"SCONJ"},{"TEXT":"ください"},{"POS":{"REGEX":"PUNCT"},"OP":"*"}]

        #matcherのコールバック関数
        #5w1hのラベルだけ付与
        def add_label(matcher, doc, id, matches):
            l = list(matches[id])
            for t in doc[l[1]:l[2]]:
                if t._._5w1h == 'None' or t._._5w1h == "Mod":
                    t._._5w1h = nlp.vocab.strings[l[0]]


        def add_right(matcher, doc, id, matches):
            l = list(matches[id])
            tag =  nlp.vocab.strings[l[0]]
            end = l[-1]

            if  end != len(doc):
                while l[1]<= doc[end].head.i <= l[-1]-1:

                    end = end + 1
                    if end == len(doc):
                        break


            l[-1] = end-1
            if end < len(doc):
                if (re.search("ので|だから",doc[l[-1]-2:l[-1]].text) or re.search("ので、|だから、",doc[l[-1]-3:l[-1]].text) or re.search("ため",doc[l[-1]-1:l[-1]].text) or re.search("ため、",doc[l[-1]-2:l[-1]-1].text)) and doc[l[-1]].pos_ != "ADP":
                    tag = "Why" 

            for t in doc[l[1]:end]:
                if t._._5w1h == 'None' or t._._5w1h == "Mod" or tag == "Why":
                    t._._5w1h = tag

            matches[id] = tuple(l)

        def add_right_left(matcher, doc, id, matches):
            l = list(matches[id])
            tag =  nlp.vocab.strings[l[0]]
            end = l[-1]
            start = l[1]

            if  end != len(doc):
                while l[1]<= doc[end].head.i <= l[-1]-1:

                    end = end + 1
                    if end == len(doc):
                        break


            if start != 0:
                while doc[start].head.i == l[1]:
                    start = start - 1
                    if start == 0:
                        break

            l[1] = start
            l[-1] = end

            if end < len(doc):
                if (re.search("ので|だから",doc[l[-1]-2:l[-1]].text) or re.search("ので、|だから、",doc[l[-1]-3:l[-1]].text) 
                    or re.search("ため",doc[l[-1]-1:l[-1]].text) or re.search("ため、",doc[l[-1]-2:l[-1]].text)) and doc[l[-1]].pos_ != "ADP":
                    tag = "Why" 

            for t in doc[l[1]:l[2]]:
                if t._._5w1h == 'None' or t._._5w1h == "Mod" or tag == "Why" :
                    t._._5w1h = tag

            matches[id] = tuple(l)
            
        def add_label_type(matcher, doc, id, matches):
            l = list(matches[id])
            for t in doc[l[1]:l[2]]:
                    t._._type = nlp.vocab.strings[l[0]]

        #matcherを追加
        matcher.add("When",add_right, When_pattern1,When_pattern2,When_pattern3,When_pattern4)
        matcher.add("Where", add_right, Where_pattern1,Where_pattern2,Where_pattern3)
        matcher.add("How", add_right, How_pattern1,How_pattern2,How_pattern3,How_pattern4,How_pattern5,
                    How_pattern6,How_pattern7,How_pattern8,How_pattern9,How_pattern10,How_pattern11,How_pattern12,How_pattern13,How_pattern14,How_pattern15)
        matcher.add("Who",add_label, Who_pattern1,Who_pattern2,Who_pattern3,Who_pattern4,Who_pattern5,
                    Who_pattern6,Who_pattern7,Who_pattern8,Who_pattern9)
        matcher.add("What", add_right, What_pattern1,What_pattern2)
        matcher.add("Why", add_label,Why_pattern1)
        matcher.add("Mod", add_right_left,Mod_pattern1,Mod_pattern2)
        
        matcher.add("Task", add_label_type,Task_pattern1,Task_pattern2,Task_pattern3,Task_pattern4)

        
        doc = nlp(text)
        text2 = [s.text for s in doc if not re.fullmatch("まあ|まぁ|ま|えー|あのー|あ",s.text) ]
        text2 = ''.join(text2)
        doc = nlp(text2)
        for sent in doc.sents:
            matches = matcher(doc)
            num = 0
            start = 0 
            end = 0

            tmp_label = None

            #その他の処理
            for token in doc:
                if (token._._5w1h == "Who" or token._._5w1h == "What") and re.search("VERB|ADJ",doc[token.head.i].pos_) and doc[token.head.i].i > token.i:
                    tag2 = "How"
                    start = token.head.i

                    end = start + 1

                    while doc[end].head.i == start and end != len(doc):

                        end = end + 1

                        if end == len(doc):
                            break

                    if (re.search("ので|だから",doc[end-2:end-1].text) or re.search("ので、|だから、",doc[end-3:end-1].text)
                        or re.search("ため",doc[end-1:end-1].text) or re.search("ため、",doc[end-2:end-1].text)) and doc[end-1].pos_ != "ADP":
                        tag2 = "Why" 


                    for t in doc[start:end]:
                        if not t._._5w1h or t._._5w1h == "Mod" or tag2 == "Why":
                            t._._5w1h = tag2


            for token in reversed(doc):

                if token._._5w1h != "Mod":
                    if tmp_label == "Who" and token._._5w1h == "What":
                        token._._5w1h = "Who"
                    tmp_label = token._._5w1h
                else:
                    token._._5w1h = tmp_label
        self.doc = doc

    def display_5w1h(self):
        doc = self.doc
        _5w1h = phrase_5w1h()
        _5w1h_list = []
        for sent in doc.sents:
            an_text = []
            an_label = ''
           # print(sent)
            _5w1h.start = sent.start
            for i,token in enumerate(sent):
                
                an_text.append(token.text)
                if i+1 < len(sent):
                    if (sent[i+1]._._5w1h != token._._5w1h):
                        #print(''.join(an_text))
                        #print(token._._5w1h)
                        #print('\n')
                        _5w1h.phrase = ''.join(an_text)
                        _5w1h.end = sent.start+i+1
                        _5w1h._type=token._._5w1h
                        _5w1h_list.append(_5w1h)
                        _5w1h = phrase_5w1h()
                        _5w1h.start=sent.start+i+1
                        

                        an_text = []
            
                else:
                    _5w1h.end = sent.end
                    _5w1h.phrase = ''.join(an_text) 
                    _5w1h._type = token._._5w1h
                    _5w1h_list.append(_5w1h)
                    _5w1h = phrase_5w1h()

 
                    #print(''.join(an_text))
                    #print(token._._5w1h)
                    #print('\n')
        return(_5w1h_list)
                    
    def display_type(self):
        doc = self.doc

        start = len(doc)-1
        while start >= 0:
            if doc[start]._._type:
                end = start+1


                while start >= 0:
                    
                    
                    if (doc[start]._._5w1h == "Who" or doc[start]._._5w1h == "What") and (doc[start].pos_ != "PRON" and doc[start].pos_ != "ADP"):
                        break
                    start = start -1
                    


                while start >= 1: 

                    if(doc[start-1]._._5w1h != doc[start]._._5w1h):
                        break
                    start = start -1



                print(doc[start:end])
                print(doc[end-1]._._type)
                print('\n')
            else:
                start = start -1
                
    def display_imp(self,imp_list):
        doc = self.doc
        
        for sent in doc.sents:
            for token in sent:
                if token.text in imp_list:
                    print(sent)
                    break
        

        