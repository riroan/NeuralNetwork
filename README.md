# NeuralNetwork
C++17로 만드는 딥러닝 라이브러리

2/22
  - FCNN구현
  
3/2
  - 속도문제로 STL vector 대신 Vector 클래스를 만들어 사용
  - 속도문제로 matrix를 vector<vector> 에서 1차원 vector로 사용
  - 코드중 잡다한 부분 수정
  - 1차원 컨볼루션레이어 구현 시작(미완성)
  
3/3
  - 1차원/2차원 컨볼루션레이어 순전파 구현

3/6
  - 2차원 컨볼루션 레이어 역전파 구현

3/7
  - 케라스같이 사용하기 유연한 모델 구현
  - 벡터, 행렬 기능 추가
  
3/12
  - 2차원 컨볼루션 레이어 역전파 오류수정(그래디언트 구하는부분 수정필요)

3/27
  - dropout, 옵티마이저 

3/29
  - 매트릭스 기능 수정
  
3/30
  - 기능 추가 (vector slice, ...)

4/2
  - 3차원 컨볼루션레이어 순전파 

4/4
  - 3차원 컨볼루션 역전파 구현
  - 행렬곱 omp를 이용해서 병렬처리
  - 맥스풀링층 구현
  - 문제점 : 메모리 비정상적으로 많이 들어감, 속도개선, 컨볼루션층에서 완전수렴 안함
  
