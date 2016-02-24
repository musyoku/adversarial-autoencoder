# -*- coding: utf-8 -*-
from activations import activations

class Config:
	def check(self):
		if self.gen_activation_type not in activations:
			raise Exception("Invalid type of activation for gen_activation_type.")
		if self.gen_output_activation_type and self.gen_output_activation_type not in activations:
			raise Exception("Invalid type of activation for gen_output_activation_type.")
		if self.dis_activation_type not in activations:
			raise Exception("Invalid type of activation for dis_activation_type.")
		if self.dis_softmax_activation_type and self.dis_softmax_activation_type not in activations:
			raise Exception("Invalid type of activation for dis_softmax_activation_type.")
		if self.dec_activation_type not in activations:
			raise Exception("Invalid type of activation for dec_activation_type.")
		if self.dec_output_activation_type and self.dec_output_activation_type not in activations:
			raise Exception("Invalid type of activation for dec_output_activation_type.")

		if config.gen_encoder_type not in {"deterministic", "gaussian"}:
				raise Exception("Invalid encoder type for gen_encoder_type.")

	def dump(self):
		pass


config = Config()

# 共通設定
config.img_channel = 1
config.img_width = 28

## Batch Normalizationを使うかどうか
config.enable_batchnorm = True

## GPUを使うかどうか
config.use_gpu = True

## 入力ベクトルの次元
config.n_x = config.img_width ** 2

## 隠れ変数ベクトルの次元
config.n_z = 100

## 重みの初期化
config.wscale_base = 0.0001

# Encoder(Generator)の設定
## xをzに符号化するEncoderとxからzを生成するGeneratorは同一のニューラルネット
## 隠れ層のユニット数
## 左から入力層側->出力層側に向かって各層のユニット数を指定
config.n_gen_hidden_units = [600, 600]

## q(z|x)の種類
## "deterministic" または "gaussian"
## 詳細は [Adversarial Autoencoder](http://arxiv.org/abs/1511.05644)
config.gen_encoder_type = "deterministic"

## 活性化関数
## See activations.py
config.gen_activation_type = "elu"

## 出力層の活性化関数
## Noneも可
config.gen_output_activation_type = None

## 出力層でBatch Normalizationを使うかどうか
## 詳細は[Deep Convolutional Generative Adversarial Networks](http://arxiv.org/pdf/1511.06434v2.pdf)
## ... This was avoided by not applying batchnorm to the generator output layer ...
## 性質上使ってしまうと隠れ変数ベクトルは平均が0分散が1になるのでp(z)に押し込めなくなると思われる
config.gen_enable_batchnorm_to_output = False

## ドロップアウト
config.gen_enable_dropout = True

# Discriminatorの設定
## 入力されたzが本物か偽物かを判断する
## 隠れ層のユニット数
## 左から入力層側->出力層側に向かって各層のユニット数を指定
config.n_dis_hidden_units = [600, 600]

## 隠れ層の活性化関数
## See activations.py
config.dis_activation_type = "elu"

## ソフトマックス層への入力の活性化関数
## Noneも可
config.dis_softmax_activation_type = "elu"

## 入力層でBatch Normalizationを使うかどうか
## 詳細は[Deep Convolutional Generative Adversarial Networks](http://arxiv.org/pdf/1511.06434v2.pdf)
## ... This was avoided by not applying batchnorm to ... the discriminator input layer.
## ラベル情報をone-hotなベクトルで与えている場合はおかしなことになるのでFalseにしておくほうが良いと思われる
config.dis_enable_batchnorm_to_input = False

## ドロップアウト
config.dis_enable_dropout = True

# Decoderの設定
## zからxを復号する
## 左から入力層側->出力層側に向かって各層のユニット数を指定
config.n_dec_hidden_units = [600, 600]

## 隠れ層の活性化関数
## See activations.py
config.dec_activation_type = "elu"

## 出力層（画素値を表す）の活性化関数
## 通常入力画像の画素値の範囲は-1から1に正規化されているためtanhを使う
## Noneも可
config.dec_output_activation_type = "tanh"

## 出力層でBatch Normalizationを使うかどうか
## 詳細は[Deep Convolutional Generative Adversarial Networks](http://arxiv.org/pdf/1511.06434v2.pdf)
## ... This was avoided by not applying batchnorm to the generator output layer ...
## 使うと出力がtanhと同じになり活性化関数を考える意味がなくなってしまうと思われる
config.dec_enable_batchnorm_to_output = False

## ドロップアウト
config.dec_enable_dropout = True