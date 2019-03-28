#
import pickle
def nlist(string,n):
    if string=='It was good talking to you!Come back anytime you wish to talk to me':
          return [0,1,string]
    if n=='2':
        return [nlist(raw_input('\n'+'sad to:'+ string+':'),raw_input('\n'+'no. of branching:')),nlist(raw_input('\n'+'happy to:'+ string+':'),raw_input('\n'+'no.of branching:')),string]

    if n=='3':
         return [nlist(raw_input('\n'+'sad to:'+ string+':'),raw_input('\n'+'no.of branching:')),nlist(raw_input('\n'+'happy to:'+ string+':'),raw_input('\n'+'no.of branching:')),nlist(raw_input('\n'+'neutral to:'+ string+':'),raw_input('\n'+'no.of branching:')),string]
        

tree=nlist('Our current memories are shaped by our past experiences.What do you think is one such expereince that has had a major impact on your personality?','3')

with open('newtree.txt','wb') as fp:
    pickle.dump(tree,fp)
