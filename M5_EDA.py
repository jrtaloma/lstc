import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed


dir_dataset = './datasets/m5-forecasting-accuracy'
num_processes = multiprocessing.cpu_count()


def process_data(i):
    sales_train_validation = pd.read_csv('./datasets/m5-forecasting-accuracy/sales_train_validation.csv')
    #sales_train_validation = sales_train_validation[sales_train_validation['dept_id']!='FOODS_3']
    #sales_train_validation = sales_train_validation[sales_train_validation['dept_id']!='HOUSEHOLD_2']
    sales = sales_train_validation.values
    sell_prices = pd.read_csv('./datasets/m5-forecasting-accuracy/sell_prices.csv')
    calendar = pd.read_csv('./datasets/m5-forecasting-accuracy/calendar.csv')[:1913]
    list_wm_yr_wk = calendar[['wm_yr_wk']].values.flatten()
    list_d = calendar[['d']].values.flatten()
    assert len(list_wm_yr_wk) == len(list_d)
    item = sales_train_validation[['id', 'item_id', 'store_id', 'cat_id']].values[i]
    rows_sales = []
    rows_sell_prices = []
    id, item_id, store_id, cat_id = item[0], item[1], item[2], item[3]
    for j,d in enumerate(list_d):
        rows_sales.append([id, item_id, store_id, cat_id, d, sales[i,6+j]])
        rows_sell_prices.append([id, item_id, store_id, cat_id, d, list_wm_yr_wk[j]])
    df_sales = pd.DataFrame(rows_sales, columns=['id', 'item_id', 'store_id', 'cat_id', 'd', 'sales'])
    df_sell_prices = pd.DataFrame(rows_sell_prices, columns=['id', 'item_id', 'store_id', 'cat_id', 'd', 'wm_yr_wk'])
    df_sell_prices = df_sell_prices.merge(sell_prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
    df = df_sales.merge(df_sell_prices, how='outer', on=['id', 'item_id', 'store_id', 'cat_id', 'd'])

    #df.to_csv(os.path.join(dir_dataset, f'{id}.csv'), index=False, na_rep=np.nan)
    return df['sales'].values.astype(np.int32), df['sell_price'].values.astype(np.float32), df['cat_id'].values

def parallel_process_data(num_processes):
    res = Parallel(n_jobs=num_processes)(delayed(process_data)(i) for i in tqdm(range(30490), total=30490))
    return res


if __name__ == '__main__':
    res = parallel_process_data(num_processes)
    res = np.array(res)
    sales, sell_prices, cat_id = res[:,0], res[:,1], res[:,2]
    cat_id = cat_id[:,0]
    del res
    print(sales.shape, sell_prices.shape, cat_id.shape)
    #np.save(os.path.join('datasets', 'm5-forecasting-accuracy', 'sales.npy'), sales)
    np.save(os.path.join('datasets', 'm5-forecasting-accuracy', 'sell_prices_individual.npy'), sell_prices)
    #np.save(os.path.join('datasets', 'm5-forecasting-accuracy', 'cat_id.npy'), cat_id)