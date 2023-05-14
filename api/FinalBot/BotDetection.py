import os
import pickle
import tweepy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


file = os.path.abspath("api\\FinalBot\\training_data_2_csv_UTF.csv")

training_data = pd.read_csv(file)
bots = training_data[training_data.bot == 1]
nonbots = training_data[training_data.bot == 0]

file2 = os.path.abspath("api\\FinalBot\\test_data_4_students.csv")
test = pd.read_csv(file2, sep='\\t', header=None)

# bots[bots.listedcount>10000]
condition = (bots.screen_name.str.contains("bot", case=False) == True) | (bots.description.str.contains(
    "bot", case=False) == True) | (bots.location.isnull()) | (bots.verified == False)
bots['screen_name_binary'] = (
    bots.screen_name.str.contains("bot", case=False) == True)
bots['location_binary'] = (bots.location.isnull())
bots['verified_binary'] = (bots.verified == False)

condition = (nonbots.screen_name.str.contains("bot", case=False) == False) | (nonbots.description.str.contains(
    "bot", case=False) == False) | (nonbots.location.isnull() == False) | (nonbots.verified == True)
nonbots['screen_name_binary'] = (
    nonbots.screen_name.str.contains("bot", case=False) == False)
nonbots['location_binary'] = (nonbots.location.isnull() == False)
nonbots['verified_binary'] = (nonbots.verified == True)

df = pd.concat([bots, nonbots])
df = df.select_dtypes(include='number')
df = df.fillna(0)  # Replace missing values with 0
df.corr(method='spearman')

# Feature Engineering
file = open(file, mode='r', encoding='utf-8', errors='ignore')
training_data = pd.read_csv(file)


def feature_eng(training_data):
    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
        r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
        r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
        r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

    training_data['screen_name_binary'] = training_data.screen_name.str.contains(
        bag_of_words_bot, case=False, na=False)
    training_data['name_binary'] = training_data.name.str.contains(
        bag_of_words_bot, case=False, na=False)
    training_data['description_binary'] = training_data.description.str.contains(
        bag_of_words_bot, case=False, na=False)
    training_data['status_binary'] = training_data.status.str.contains(
        bag_of_words_bot, case=False, na=False)
    return training_data


training_data = feature_eng(training_data)

# Feature Extraction


def feature_ext(training_data):
    training_data['listed_count_binary'] = (
        training_data.listed_count > 20000) == False
    features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary',
                'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']
    return training_data


training_data = feature_ext(training_data)

consumer_key = '25SjEQNdimGLs9BNcAfbJW3dA'
consumer_secret = 'RTt7e2m4iWwbXUUHyH4Vn7YRm6jpoQmm4m8RhedqohQBNbyYLU'
access_key = '755246834826838016-GPchEozsoRFTm10LbSbUKyG2NlIoLOR'
access_secret = 'x0LXflU8vJFojsXfgumxLNlh8TEMUCUpqkK5fuH98UY6o'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def getname(name):
    _id = name
    user = api.get_user(screen_name=_id)
    user.status = ''

    checking = [[user.id, user.id_str, user.screen_name, user.location, user.description, user.url, user.followers_count, user.friends_count, user.listed_count, user.created_at,
                 user.favourites_count, user.verified, user.statuses_count, user.lang, user.status, user.default_profile, user.default_profile_image, user.has_extended_profile, user.name]]
    check_df = pd.DataFrame(checking, columns=['id', 'id_str', 'screen_name', 'location', 'description', 'url', 'followers_count', 'friends_count', 'listed_count',
                                               'created_at', 'favourites_count', 'verified', 'statuses_count', 'lang', 'status', 'default_profile', 'default_profile_image', 'has_extended_profile', 'name'])
    checking2 = feature_eng(check_df)
    checking2 = feature_ext(check_df)

    features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary',
                'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']
    features1 = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary',
                 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary']
    X = training_data[features].iloc[:, :-1]
    y = training_data[features].iloc[:, -1]
    z = checking2[features1].iloc[:, :]

    # DecisionTree Model
    with open('C:\\Users\\Irtaza\\Desktop\\project\\api\\FinalBot\\DTModel', 'rb') as f:
        mp = pickle.load(f)

    z_pred_test = mp.predict(z)

    return z_pred_test
