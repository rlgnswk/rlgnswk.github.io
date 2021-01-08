## Test Posting
**제목: Second-order Attention Network for Single Image Super-Resolution**

**(CVPR19)**

[daitao/SAN](https://github.com/daitao/SAN)

**Intro**

기존 SR-CNN 모델들은 그냥 깊게, 넓게만 만들려는 경향이 있다고 지적하면서 2가지 한계를 말한다.

1. input LR 이미지의 정보를 전부 사용하고 있지않다.
2. high-level features를 추출하려고만 하고 inherent feature correlations in intermediate layers를 추출하지 않는다.

따라서, 본 논문은 다음과 같이 제시한다.(summery)

1. propose a deep second-order attention network(SAN) for accurate image SR
2. propose second-order channel attention (SOCA)
3. propose non-locally enhanced residual group(NLRG) structure to build a deep network

**Attention mechanism**

각 채널의 중요도를 계산 (1x1xC) 벡터를 만들어서 기존 (HxWxC)벡터의 각 채널별로 곱해준다(가중치를 준다).  ← 중요한 정보를 강조한다 ← Attention

연산은 이해가 가는데 수학적으로 어떻게 성능이 높여지는 건지는 이해가 가지않음.. 

SENet에서 처음 제안

[SENet(Squeeze and excitation networks)](https://jayhey.github.io/deep%20learning/2018/07/18/SENet/)

하지만 이 방법은 first-order한 방법이라서 더 고차원 적인 통계를 무시하여 네트워크의 성능을 방해한다고 나와있고 SR task에서는 그러한(?) high frequency(=고차원정보?)가 더 중요하다고 말한다.   

> """
However, SENet only
explores first-order statistics (e.g., global average pooling),
while ignoring the statistics higher than first-order, thus hindering the discriminative ability of the network. Image SR, features with more high-frequency information are more informative for HR reconstruction.
"""

**Second-order Attention Network (SAN)**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/648178e7-18f9-4a61-beba-5156b8359292/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/648178e7-18f9-4a61-beba-5156b8359292/Untitled.png)

네트워크 전체 구조 및 모듈

**Loss function**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d575805c-2e6b-434f-818c-050bbaace5eb/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d575805c-2e6b-434f-818c-050bbaace5eb/Untitled.png)

L1 loss를 사용했다.

**Non-locally Enhanced Residual Group (NLRG)**

모델 가운데를 차지하는 NLRG는 conv layer하나와 RL_NL module로 뽑힌 shallow feature가 매 모듈마다 더해지는데 Wssc 라는 학습 가능한 가중치가 붙는다. 

보여지는 점으로는 Dense 한 연결이 아닌 그냥 residual한 연결인점..?

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/74d100d9-93ec-4b0b-8e36-73b5f1ecc37b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/74d100d9-93ec-4b0b-8e36-73b5f1ecc37b/Untitled.png)

**Region-level non-local module (RL-NL)**

SSRG 앞뒤로 붙어있는 모듈이다. 

우선 non local neural network가 뭔지 이해를 해야한다.

[Non-local Neural Networks](https://blog.lunit.io/2018/01/19/non-local-neural-networks/)

CNN에서 pooling 이나 conv 연산은 모두 receptive field에 종속되는 local operation이다. 이와 다르게 non-local한 operation을 사용하면 기존 다른 특징들을 알아낼 수 있게 된다.

논문에서는 기존 non-local operation의 두가지 취약점을 말하면서 Region level로 나눠야 하는 이유를 설명한다.

1. 용량이 많이 필요함
2. 경험적으로 super resolution같은 low level task(인용한 논문을 봐야지 이해 될듯)은 적절한 근처의 크기가 바람직함

그래서 한 이미지를 grid하게 잘라 region level로 만들고 그 안에서 non local operation이 실행되게 만들었다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/53fcc1e2-e73e-466f-8d31-6c3989b1e487/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/53fcc1e2-e73e-466f-8d31-6c3989b1e487/Untitled.png)

**Local-source residual attention group (LSRAG).**

LSRAG은 그 안에 conv + ReLU,  그리고 attention모듈(SOCA)까지 합쳐져서 한 블록으로 모델 구조상 블록간 cascade 된다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4a1d00b3-76aa-4b1c-81bb-0e14d7b93165/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4a1d00b3-76aa-4b1c-81bb-0e14d7b93165/Untitled.png)

**Second-order Channel Attention (SOCA)**

논문에서는 기존 SR 논문에서는 feature 간 interdependencies를 고려하지 않았다고 말한다. 또한 SEnet에서는 first order information만 고려했다고 하는데, 다른 second order information을 고려한 연구들을 제시하면서 seconde order information을 포함한 attention 기법을 제시한다.

이 페이지 부터 수식이 많이 나와서 정확히 이해한지는 모르겠습니다..

**Covariance normalization**

다음과 같은 시그마를 X의 covariance normalization이라고 한다고 한다. 여기서 X를 CxHxW로 봐야지 말이되는거같다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fbf8efa7-3bac-4875-ad06-0ad39921f850/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fbf8efa7-3bac-4875-ad06-0ad39921f850/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21438c36-c252-4d67-8fc1-6d27bd7e6f26/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21438c36-c252-4d67-8fc1-6d27bd7e6f26/Untitled.png)

이 시그마는 symmetric positive semi-definite 해서 eigenvalue decomposition할 수 있다고 말한다.

그래서 아래와 같이 Λ =diag(λ1, ··· , λC )를 만든다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/31eb123b-363a-480c-b793-b0d4efd37324/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/31eb123b-363a-480c-b793-b0d4efd37324/Untitled.png)

근데 이걸 다음과 같이 지수적으로 나타낼 수 있는데(Λ^α =diag(λ_1^α, ··· , λ_C^α ))

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c2bbb88-f55f-4b8d-9169-73b1698df115/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c2bbb88-f55f-4b8d-9169-73b1698df115/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61bb41e3-ee9e-41dc-aa75-1ff81f62e37e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61bb41e3-ee9e-41dc-aa75-1ff81f62e37e/Untitled.png)

이떄 α를 1보다 작은 수로 둠으로써 1보다 큰 eigenvalue는 감소하고, 1보다 작으면 증가하게 만들수 있다 → 강조할 수 있다. 논문에서는 α=1/2로 두었다.

**Channel attention**

위에 말한 Y_hat을 이용하여 1x1xC를 만드는데,  global covariance pooling이라는 operation을 통해서 이루어진다. 먼저 위의 수식을 통해 Y_hat을 만들고 아래 수식을 통해 Matrix의  channel-wise statistics을 구할수 있다고 한다.

이거 수식이 맞는가 싶은데 밑에 내가 그린걸로 되야지 말이 되는거 아닌가..???

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11186720-fc6d-475b-9b52-f7d330691029/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11186720-fc6d-475b-9b52-f7d330691029/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c72aef90-639d-463a-9bb4-3fb42d02e019/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c72aef90-639d-463a-9bb4-3fb42d02e019/Untitled.png)

하여튼 그 이후로는 아까 **Attention mechanism**에서 설명한 연산을 통해 feature map을 down -sampling 후 up-sampling하는 연산을 통해 각 채널간 가중치를 구하고 기존 feature map에 곱해줌으로서 SOCA 모듈을 구성했다.

**Covariance Normalization Acceleration**

이 부분은 Newton-Schulz iteration 라는 것을 통해 Covariance Normalization matrix구하는 연산을 빠르게 했다는거 같은데, 논문 전체라기보단 어려운 디테일인거 같아서 패스 하겠슴다.

**Implementations**

RL-NL modules 에서는 k=2 로 해서 k^2 = 4등분을 했다.

LSRAG 에서는10 residual blocks + SOCA 로 구성했고

SOCA 모듈에서 1 × 1 convolution filter with reduction ratio r = 16(fc 대체) 을 사용했다.

다른 conv filter는 3x3이고 채널은 C=64로 고정인듯 하다.

upsampling은 bi-linear up-sampling을 사용했다고 다른 포스팅에서 봤다.

**Discussions**

여기서는 신기하게?도 다른 모델들과 비교하는데, 다른 모델을 까버리고, 논문에 제시한 부분을 높이는데 이렇게 글을 써도 되나 싶다ㅋㅋ

다른거는 읽지 않았고, RCAN과의 차이점을 봤는데, 저자도 비슷하다고는 언급했다. 하지만 3가지 차이가 있다. 

1. RCAN은 LSC을 이용하여 마지막에 f0 feature map을 더해줬는데, SAN은 매 단계 share-source skip connections을 통해 더해줬다는 점
2. RCAN은 local operation만 썼다는점
3. RCAN의 attention은 first order라는 점을 든다.

**이후**

이후에는 성능비교와 사진들, 그리고 코멘트들 있는데 빠르게 넘겼다.

그중에서 파라미터를 비교한게 있는데 다음과 같이 나왔다. RCAN과 비슷한 수준이며 준수하지않나? 싶다. DRLN에서 나온 표를 봐도 RCAN은 중간정도 위치해있다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38225c3f-7273-4383-ae17-185c07b8aa2f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38225c3f-7273-4383-ae17-185c07b8aa2f/Untitled.png)

**Comment**

DRLN하고 궁금해서 비교해보니 4x기준으로 성능이 엎치락 뒤치락 한다. 

Laplacian attention이냐 second order attention이냐 차이인거같기도하고,

RCAN, DRLN은attention을 엄청 자주 한 residual block당 한번 꼴이고, SAN은 여러 block당 한번 SOCA를 적용한 차이일 수도 있을거 같다.
또 이쯤되니 attention기법이 무조건 나오는거 같은데 GAN이도 적용이 되나 싶다. 
되면 내 논문 제목은 Attention - SRGAN으로 ASRGAN이다.

신나서 검색했는데 이미 있다;


