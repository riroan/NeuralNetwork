# NeuralNetwork
C++로 만드는 뉴럴 네트워크
구현환경 : visual studio (C++17)

2/22
  - FCNN구현
  
3/2
  - 속도문제로 STL vector 대신 Vector 클래스를 만들어 사용
  - 속도문제로 matrix를 2차원 vector 에서 1차원 vector로 사용
  - 코드중 잡다한 부분 수정
  - 1차원 컨볼루션레이어 구현 시작(미완성)
  
3/3
  - 1차원/2차원 컨볼루션레이어 순전파 구현
  
3/7
  - 2차원 컨볼루션레이어 역전파 구현(풀링 미구현, 패딩/스트라이드 미적용)
  - 신경망을 더 유연하게 사용하기 위해서 모델클래서 1차 구현(케라스 같이 사용가능)
  - 벡터 행렬 기능 추가
