from IPython import get_ipython
get_ipython().magic('reset -sf')


# amazon data extract
import pandas as pd
import gzip
import itertools

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
    

def getDF(path, start_idx, end_idx):
    i = 0
    df = {}
    chunk = itertools.islice(parse(path), start_idx, end_idx)
    for d in chunk:
        #print(d['reviewerID'])
#        print(d['overall'] == class_score)
        df[i] = d
        i += 1

    return pd.DataFrame.from_dict(df, orient='index')


def main():
    counters = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0}
    for chunk_idx in range(1, 10000):
        start_idx = 10000 * chunk_idx
        end_idx = (chunk_idx+1) * 10000
        df = getDF('item_dedup.json.gz', start_idx, end_idx)
        df2 = df.filter(items=['overall', 'reviewText', 'summary'])
        
        for score, sub_df in df2.groupby('overall'):
            print 'counter for ', score, ' is ', counters[score]
            ready = all(value > 750000 for value in counters.values())
            if ready is True:
                return
            if counters[score] > 750000:
                continue
            else:
                df_score = pd.DataFrame(columns=['overall', 'reviewText', 'summary'])
                try:
                    sub_draw = sub_df.sample(300) # 730000
                    df_score = df_score.append(sub_draw)
                    counters[score] = counters[score] + sub_draw.shape[0]
            
                    sub_draw.to_csv (r'cheryl_amazon_data_' + str(score) + '.csv',
                                    mode='a', index = None, header=True)
                    print "write " , str(sub_draw.shape), ' to ',  str(score)
                except:
                    print "problem " , str(sub_draw.shape), ' to ',  str(score)
        

main()