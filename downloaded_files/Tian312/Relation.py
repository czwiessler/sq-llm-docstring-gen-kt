__author__ = 'Tian Kang'

#============ Parser step 2: clincial relation identification =====#
#                                                                  #
#   arguement 1:  xml output dir                                   #
#   arguement 2: file name
#                                                                  #
#   email: tk2624@cumc.columbia.edu (Tian)                         #
#   June, 2016                                                     #
#                                                                  #
#==================================================================#
import os,codecs,re
import sys
from libsvm import svmutil
from features_dir import relation_features
from xml.etree import ElementTree as ET
import sys
def main():
    m=svmutil.svm_load_model('trained_models/svm.model')
    relation_tag={0:'None',1:'has_value',2:'has_temp',3:'modified_by'}

    match=re.search('^(.*)\.txt',sys.argv[2])
    filename=sys.argv[2]
    if match:
        filename=match.group(1)

    input_dir=sys.argv[1]+'/'+filename+'_NER.xml'
    output_dir=sys.argv[1]+'/'+filename+'_Parsed.xml'
    print "Reading NER results from ", input_dir

    tree = ET.ElementTree(file=input_dir)

    root = tree.getroot()
    relations={}
    index=[]
    for child in root:

        syn_features=codecs.open('Tempfile/relation_scale','w')
        temp_pairs=relation_features.generate_pairs(child,syn_features)
        if temp_pairs:
            try:
                y,x=svmutil.svm_read_problem('Tempfile/relation_scale')
                p_label,p_acc,p_val=svmutil.svm_predict(y,x,m)
            except ValueError:
                for child2 in child.findall('text'):
                    print child2.text
                continue

        #print len(p_label),len(temp_pairs)
        else:
            p_label=[]
            temp_pairs=[]
        for j in range(0,len(p_label)):
            #print j
            relations[temp_pairs[j]]=p_label[j]

            indexes=temp_pairs[j].split("_")
            index.append(indexes[0])
            index.append(indexes[1])

        for child2 in child.findall('entity'):
            node_index=child2.attrib['index']
            child2.attrib['relation']='None'

            if node_index in index:

                right_pattern='^(\w+)_'+node_index
                left_pattern=node_index+'_(\w+)$'
                for relation in relations:
                    match1=re.search(left_pattern,relation)
                    match2=re.search(right_pattern,relation)
                    other_index= None
                    if match1:
                        other_index=match1.group(1)
                    else:
                        if match2:
                            other_index=match2.group(1)
                        else:
                            continue
                    relation_type=relation_tag[relations[relation]]
                    if relation_type == 'None':
                        continue

                    if child2.attrib['relation'] is 'None':
                        child2.attrib['relation']=other_index+":"+relation_type
                    else:
                        child2.attrib['relation']=child2.attrib['relation']+"|"+other_index+":"+relation_type
            #print child2.text,child2.attrib['index'],child2.attrib['relation']

    relation_excuted=os.path.exists("in.parse")
    if relation_excuted:
        os.system('rm in.parse')
    os.system('rm Tempfile/relation_scale')
    print "Writing Relation xml to ", output_dir
    new_tree=codecs.open(output_dir,'w')
    tree.write(new_tree)
    print "Finished!"

if __name__ == '__main__': main()
