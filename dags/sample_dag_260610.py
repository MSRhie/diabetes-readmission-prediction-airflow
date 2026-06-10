#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Example Airflow DAG that shows the complex DAG structure.
"""

from __future__ import annotations

import pendulum

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    # hostlocal 의 DAGS에 보이는 이름, 파이썬파일명과는 관련없음, 단 파일명과 일치시키는게 좋음(파일들이 많아지므로)
    dag_id="dags_bash_operator",
    # 매일 0시 0분에 실행 # 분 시 일 월 요일이란 뜻
    schedule="0 0 * * *",
    # DAG 시작일자 설정(서울시간 기준) 이떄 catchup 변수와 같이 봐야함
    start_date=pendulum.datetime(2021, 1, 1, tz="Asia/Seoul"),
    # 이전날짜 만약 3월에 DAG 시작일자가 되어있다면, 1월과 3월사이의 작업이 한번에 돌아감(문제 발생할수있으므로 주의)
    # 일반적으로 catchupd=False 로 설정함
    catchup=False,
    # hostlocal 의 DAGS에 보이는 태크를 의미
    tags=["example", "example2", "example3"],
    # DAG 실행 최대 시간 설정
    # dagrun_timeout=timedelta(minutes=60),  
    # 아래 DAG 내에서 공통으로 사용할 변수 설정
    # params={"example_key": "example_value"},
) as dag:
    # task1(run_this), task2, task3 는 BashOperator 클래스를 이용하여 생성된 태스크 객체
    bash_t1 = BashOperator(
        # task_id 는 hostlocal 의 Tasks에 보이는 이름(라벨링), task명과 동일하게 적어주는게 더 좋음
        task_id='bash_t1',
        # echo는 리눅스 명령어, whoiamI 는 출력할 문자열
        bash_command='echo whoiamI',
    )

    bash_t2 = BashOperator(
        # task_id 는 hostlocal 의 Tasks에 보이는 이름(라벨링), task명과 동일하게 적어주는게 더 좋음
        task_id='bash_t2',
        # WSL 설치된 터미널이름을 출력함
        bash_command='echo $HOSTNAME',
    )

    # task의 실행순서를 정의

    bash_t1 >> bash_t2