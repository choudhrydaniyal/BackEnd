from api.models import Post, Profile
from django.contrib.auth.models import User
from rest_framework import serializers
from djoser.serializers import UserCreateSerializer
from django.contrib.auth import get_user_model

User=get_user_model()

class UserCreateSerializer(UserCreateSerializer):
    class Meta(UserCreateSerializer.Meta):
        model=User
        fields=('id','username','email','password','first_name','last_name')


class PostCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model=Post
        fields='__all__'

class ProfileCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model=Profile
        fields='__all__'