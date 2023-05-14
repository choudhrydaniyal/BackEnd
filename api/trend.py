

import tweepy
import pandas as pd

def get_trends():
    consumer_key='25SjEQNdimGLs9BNcAfbJW3dA'
    consumer_secret='RTt7e2m4iWwbXUUHyH4Vn7YRm6jpoQmm4m8RhedqohQBNbyYLU'
    access_key = '755246834826838016-GPchEozsoRFTm10LbSbUKyG2NlIoLOR'
    access_secret = 'x0LXflU8vJFojsXfgumxLNlh8TEMUCUpqkK5fuH98UY6o'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    test_list = []
    import geocoder
    g = geocoder.ip('me')
    print(g.latlng)

    get=api.closest_trends(lat=g.latlng[0],long=g.latlng[1])

    cid=get[0].get('parentid')

    trands=api.get_place_trends(id=cid)
    trends=trands[0].get('trends')
    print(trands[0].get('trends')[0].get('name'))


get_trends()
