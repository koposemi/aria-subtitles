# ARIA 모바일 자막

폰·태블릿에서 영상을 틀면, 그 영상의 **소리만** 뽑아 Groq(클라우드)로 일본어를 인식·한국어로 번역하고
**영상 위에 자막**을 띄웁니다. 서버가 기기 안(Termux)에서 돌아서 PC가 필요 없습니다.

```
Firefox + Tampermonkey 스크립트  ──127.0.0.1:8765──▶  Termux 안의 자막 서버
   (영상 위 자막 표시)                                  (yt-dlp/ffmpeg로 소리 추출 → Groq 인식·번역)
```

## 처음 설치 (기기마다 한 번)

필요한 것: **Termux**(GitHub/F-Droid 버전, Play 스토어 버전 아님), **Firefox**, Firefox 애드온 **Tampermonkey**, Groq API 키.

이 저장소는 **비공개**라서 Termux에서 GitHub 로그인이 먼저 필요합니다.

```bash
termux-setup-storage            # 저장소 권한 허용 (키 파일을 다운로드 폴더에서 가져올 때 필요)
pkg install -y gh git           # GitHub 도구
gh auth login                   # GitHub.com → HTTPS → "Login with a web browser" → 화면의 코드를 브라우저에 입력
gh auth setup-git
gh repo clone koposemi/aria-subtitles ~/aria
cd ~/aria && bash setup.sh
```

`setup.sh`가 하는 일: 프로그램 설치(python, numpy, ffmpeg, nodejs, yt-dlp) → Groq 키 복사 → `aria` 명령 생성 →
Termux를 열면 서버가 자동으로 켜지게 설정.

### Groq 키 넣기

키는 저장소에 **올라가 있지 않습니다**(`.gitignore`로 막아 둠). 둘 중 하나로 넣으세요.

- 키를 한 줄에 하나씩 적은 `groq_key.txt`를 기기의 **다운로드 폴더**에 넣고 `setup.sh` 실행 → 자동으로 복사하고 원본은 지웁니다.
- 또는 직접 `~/.config/realtime-subtitles/groq_key.txt` 에 한 줄에 하나씩 적기.

키 여러 개를 넣으면 한도가 차는 대로 다음 키로 넘어갑니다.

### 브라우저 스크립트 설치

Termux에서 서버가 켜진 상태로 Firefox에서 아래 주소를 열면 Tampermonkey가 설치 화면을 띄웁니다.

```
http://127.0.0.1:8765/aria.user.js
```

### 꼭 해 둘 것

- 배터리 설정에서 **Termux = 제한 없음** (안 하면 화면을 끄면 서버가 죽을 수 있음).
- 서버가 켜져 있으려면 Termux가 백그라운드에 살아 있어야 합니다.

## 업데이트

```bash
cd ~/aria && git pull
```

서버가 켜져 있었다면 Termux를 완전히 닫았다 다시 열어 주세요. (프로그램 목록이 바뀐 업데이트일 때만 `bash setup.sh` 한 번 더.)

## 사용

영상 페이지에서 재생하면 서버가 소리를 받아 자막을 만들고, 몇 초 뒤부터 영상 위에 한국어 자막이 나옵니다.
화면의 **ON/OFF 버튼**으로 자막을 끄고 켭니다(끄면 서버 작업도 멈춤).
다른 사이트에서 쓰려면 `youtube/aria-youtube.user.js` 맨 위의 `// @match` 줄을 추가하세요.

## 구성

| 경로 | 설명 |
|---|---|
| `setup.sh` / `start.sh` | Termux 설치 / 서버 시작 |
| `src/realtime_subtitles/youtube_subs/server.py` | 자막 서버 (소리 추출, 30초 단위 처리, 캐시, HTTP API) |
| `src/realtime_subtitles/youtube_subs/groq_client.py` | Groq 인식·번역, 키 순환, 한도 대기 |
| `src/realtime_subtitles/youtube_subs/av_glossary.py` | 번역 용어집 |
| `youtube/aria-youtube.user.js` | Tampermonkey 스크립트 (영상 위 자막, ON/OFF 버튼) |

## 환경 변수 (필요할 때만)

| 이름 | 기본 | 뜻 |
|---|---|---|
| `ARIA_PORT` | 8765 | 서버 포트 |
| `ARIA_TARGET_LANG` | kor_Hang | 번역 대상 언어 |
| `ARIA_GLOSSARY` | match | 용어집: `match`(나온 단어만) / `20` / `30` / `50`(전체) / `off` |
| `ARIA_LAN` | 1 (`start.sh`는 0) | 1이면 같은 와이파이의 다른 기기도 접속 (토큰 필요) |
| `ARIA_CLEAR_SUBS` | 꺼짐 | 켜면 자막 캐시를 재생 후 바로 지움 (기본은 3일 보관) |

자막 캐시는 `~/aria/cache/youtube/` 에 3일간 보관됩니다. 로그는 `~/aria/logs/`.
