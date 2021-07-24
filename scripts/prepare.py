import warnings

import pandas as pd


def loadData():
    warnings.filterwarnings('ignore')
    data_result = pd.read_csv('../data/AdSmartABdata.csv')

    return data_result


def drop_col_noresponse(df):
    impression_data = df.query("not (yes == 0 & no == 0)")
    return impression_data


def split_data(df):
    try:

        v_browser = df[df.columns[~df.columns.isin(['platform_os'])]]
        v_platform = df[df.columns[~df.columns.isin(['browser'])]]

        return v_browser, v_platform
    except KeyError as e:
        print("key columns is missing")
        return df


def main():
    campaign = loadData()
    impression_data = drop_col_noresponse(campaign)
    impression_data['target'] = impression_data['yes'].map(lambda x: x == 1)
    impression_data = impression_data.drop(columns=['yes', 'no', 'auction_id'], axis=1)
    v_browser, v_platform = split_data(impression_data)
    v_browser.to_csv(r'../data/v_browser.csv', index=False, header=True)
    v_platform.to_csv(r'../data/v_platform.csv', index=False, header=True)


if __name__ == '__main__':
    main()
