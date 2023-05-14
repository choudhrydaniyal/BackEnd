
from typing import Text
from django.http.response import HttpResponse
from rest_framework.views import APIView
from rest_framework import serializers, status
from rest_framework.response import Response
from .serializers import PostCreateSerializer, ProfileCreateSerializer
from rest_framework.decorators import api_view
import json

from rest_framework.permissions import AllowAny

from rest_framework import viewsets
from .FinalBot.BotDetection import getname
from .detection import detecting_fake_news
from .models import Post, Profile


import tweepy
import pandas as pd


def gettrends():
    consumer_key = '25SjEQNdimGLs9BNcAfbJW3dA'
    consumer_secret = 'RTt7e2m4iWwbXUUHyH4Vn7YRm6jpoQmm4m8RhedqohQBNbyYLU'
    access_key = '755246834826838016-GPchEozsoRFTm10LbSbUKyG2NlIoLOR'
    access_secret = 'x0LXflU8vJFojsXfgumxLNlh8TEMUCUpqkK5fuH98UY6o'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    import geocoder
    g = geocoder.ip('me')
    if g:
        get = api.closest_trends(lat=g.latlng[0], long=g.latlng[1])
    else:
        get = api.closest_trends(lat=33, long=73)

    cid = get[0].get('parentid')
    trands = api.get_place_trends(id=cid)
    li = []
    for n in trands:
        for a in n.get("trends"):
            if str(a.get("tweet_volume")) != "None":
                a.pop('url')
                a.pop('promoted_content')
                a.pop('query')

                li.append(a)

    trend = json.dumps(li)
    val = eval(trend)

    return val


class get_profile(APIView):
    def post(self, request, format=None):
        # print("request data====",request.FILES["file"])
        profile_serializer = ProfileCreateSerializer(data=request.data)
        print(profile_serializer)
        if profile_serializer.is_valid():
            profile_serializer.save()
            return Response(profile_serializer.data, status=status.HTTP_201_CREATED)

    def get(self, request, *args, **kwargs):
        if request.method == 'GET':
            user = request.user
            posts = Profile.objects.filter(user=user)
            serializer = ProfileCreateSerializer(posts, many=True)
            return Response(serializer.data)


class get_trends(APIView):
    def get(self, request):
        item = gettrends()
        return Response(item, status=status.HTTP_200_OK)


class Postdata(APIView):

    def post(self, request):
        name = request.data['name']
        text = request.data['text']
        y = detecting_fake_news(text)
        request.data['text_result'] = y
        try:
            x = getname(name)
            if x == [1]:
                request.data['name_result'] = False
            else:
                request.data['name_result'] = True
        except:
            print("")
            request.data['name_result'] = False
        serializer = PostCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            user = request.user
            items = Post.objects.filter(Username=user).order_by('-id')
            serializer = PostCreateSerializer(items, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, id=None):
        if id:
            item = Post.objects.get(id=id)
            item.delete()
            user = request.user
            items = Post.objects.filter(Username=user).order_by('-id')
            serializer = PostCreateSerializer(items, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        user = request.user
        items = Post.objects.filter(Username=user).order_by('-id')
        serializer = PostCreateSerializer(items, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ProfileViewSet(APIView):
    def post(self, request, *args, **kwargs):
        image = request.data['name']
        user = request.data['id']
        print("*******************************************")
        Profile.objects.create(image=image, user_id=user)
        user = request.user
        posts = Profile.objects.filter(user=user)
        serializer = ProfileCreateSerializer(posts, many=True)
        return Response(serializer.data, status=200)
