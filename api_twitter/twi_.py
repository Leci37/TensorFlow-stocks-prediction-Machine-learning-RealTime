from datetime import datetime

import tweepy
# Authenticate to Twitter
from LogRoot.Logging import Logger
from api_twitter import twi_credential
#https://realpython.com/twitter-bot-python-tweepy/
# auth = tweepy.OAuthHandler(twi_credential.API_Key, twi_credential.API_Key_Secret)
# auth.set_access_token(twi_credential.Access_Token,twi_credential.Access_Token_Secret )
#
# api = tweepy.API(auth)
# client = tweepy.Client(consumer_key=twi_credential.API_Key,
#                     consumer_secret=twi_credential.API_Key_Secret,
#                     access_token=twi_credential.Access_Token,
#                     access_token_secret=twi_credential.Access_Token_Secret)
# auth = tweepy.OAuth1UserHandler(
#     twi_credential.API_Key,
#     twi_credential.API_Key_Secret,
#     twi_credential.Access_Token,
#     twi_credential.Access_Token_Secret
# )
#
# api = tweepy.API(auth)


# TWE_TEXT_EX ="""📈 BUY 📈: TEST_STOCK 2022-12-12
# Value: 26.63
# Investing.com
# WeBull.com
#
# Confidence of models:
#     POS_score: 112.9%/3
#     NEG_score: 0.0%/3
# 📊Names:
#  RIVN_pos_low1_mult_dense2: 89%
#  RIVN_pos_reg4_mult_dense2: 96%
#  RIVN_pos_reg4_mult_128: 91%"""






# make_tweet("TEST_STOCK "+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# try:
#     api.verify_credentials()
#     print("Authentication OK")
# except:
#     print("Error during authentication")

# def twitter_api_v1():
#     global TW_API_CREDENT
#     if TW_API_CREDENT is None:
#         consumer_key = twi_credential.API_Key # os.getenv('TWITTER_CONSUMER_KEY')
#         consumer_secret = twi_credential.API_Key_Secret # os.getenv('TWITTER_CONSUMER_SECRET')
#         access_token = twi_credential.Access_Token #os.getenv('TWITTER_ACCESS_TOKEN')
#         access_token_secret = twi_credential.Access_Token_Secret #os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
#
#         auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#         auth.set_access_token(access_token, access_token_secret)
#
#         TW_API_CREDENT = tweepy.API(auth)
#         Logger.logr.info("\nLOAD twitter_api \tAPI.user_agent: "+ str( TW_API_CREDENT.user_agent) )
    #     return api_tw
    # else:
    #     return api_tw


import tweepy
TW_API_CREDENT = None
API_client = None
API_auth = None
#https://stackoverflow.com/questions/76810100/how-do-i-fix-403-forbidden-error-when-using-tweepy-to-access-the-twitter-api/76810679#76810679
def twitter_api_v2():
    global API_client, API_auth, TW_API_CREDENT
    if TW_API_CREDENT is None or API_client is None:
        API_client = tweepy.Client(
            consumer_key=twi_credential.API_Key,
            consumer_secret=twi_credential.API_Key_Secret,
            access_token=twi_credential.Access_Token,
            access_token_secret=twi_credential.Access_Token_Secret,
        )
        API_auth = tweepy.OAuth1UserHandler(
            twi_credential.API_Key,
            twi_credential.API_Key_Secret,
            twi_credential.Access_Token,
            twi_credential.Access_Token_Secret
        )

        # Create API object
        TW_API_CREDENT = tweepy.API(API_auth, wait_on_rate_limit=True)
        Logger.logr.info("\nLOAD twitter_api \tAPI.user_agent: " + str(TW_API_CREDENT.user_agent))



#Put your Bearer Token in the parenthesis below
# client = tweepy.Client(bearer_token=twi_credential.Bearer_Token)
def create_simple_tweet(text):
    global TW_API_CREDENT
    twitter_api_v2()
    tweet = TW_API_CREDENT.update_status(status = text)
    return tweet


# Replace the text with whatever you want to Tweet about
# path_tech = r"C:\Users\Luis\Desktop\LecTrade\LecTrade\plots_relations\Trader_View_png\AMD_TRAVIEW_tech.png"
# path_finan =  r"C:\Users\Luis\Desktop\LecTrade\LecTrade\plots_relations\Trader_View_png\AMD_TRAVIEW_finan.png"
def put_tweet_with_images(text, list_images_path:list, in_reply_to_status_id = None):
    global TW_API_CREDENT
    twitter_api_v2()
    # Logger.logr.debug("Tweet will Send  " + str(text) )

    # try:
    list_media = []
    for i in list_images_path:
        list_media.append(TW_API_CREDENT.media_upload(filename=i).media_id)
    print("MEDIA: images found :", len(list_media), " Media_id Strigs: ", ', '.join(str(v) for v in list_media))

    # Post tweet with image
    tweet_sent = API_client.create_tweet(text=str(text).encode('utf-8', errors='replace').decode('utf-8'), media_ids=list_media)
    Logger.logr.info("Tweet has sent  tweet_ID_str: " + tweet_sent.data['id'] ) #+ " Text:" + tweet_sent.data['text'])
    # if in_reply_to_status_id is None: # .encode('utf-8', errors='replace').decode('utf-8')
    #     tweet_sent = TW_API_CREDENT.update_status(status=str(text).encode('utf-8', errors='replace').decode('utf-8'), media_ids=list_media)
    # else:
    #     tweet_sent = TW_API_CREDENT.update_status(status=str(text).encode('utf-8', errors='replace').decode('utf-8'), media_ids=list_media, in_reply_to_status_id = in_reply_to_status_id)

    return tweet_sent
    # except Exception as ex:
    #     print("Error during authentication ", ex)

# tweet_sent = put_tweet_with_images("'\033[1m  Your Name  \033[0m'" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), [ path_tech, path_finan])
# tweet_sent = put_tweet_with_images("TEST_STOCK " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), [path_tech, path_finan])

# api = twitter_api()
#
# # put_tweet_with_images("TEST_STOCK " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), [path_tech], in_reply_to_status_id=tweet_sent.id_str)
# # upload the file
# media = api.media_upload(filename=path_tech)
# # printing the information
# print("The media ID is : " + media.media_id_string)
# print("The size of the file is : " + str(media.size) + " bytes")
#
# # media = api.media_upload(filename="./assets/twitter-logo.png")
# # print("MEDIA: ", media)
#
# tweet = api.update_status(status="Image upload", media_ids=[media.media_id_string])
# print("TWEET: ", tweet)
#
#
# url = f"https://www.iheartradio.ca/image/policy:1.15731844:1627581512/rick.jpg?f=default&$p$f=20c1bb3"
# # path = r'C:\....\...\....\.......\images.jpg'
# # path = 'images.jpg'
# #
# text = "Youpi It works!"
# #
# get_picture(url, path_tech)
#https://stackoverflow.com/questions/71701199/upload-media-on-twitter-using-tweepy-with-python
# twitter_api().client().create_tweet("HOLAsss")
# twitter_api().update_status_with_media(text, path_tech)