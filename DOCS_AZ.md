# AI Köməkçi V4 — Tam Sənədləşmə (A-dan Z-yə)

> Bu sənəd "AI Köməkçi V4" sisteminin **bütün** funksional və texniki tərəflərini Azərbaycan dilində izah edir. Sistemə yeni başlayan istifadəçilər və onu inkişaf etdirəcək developerlər üçün hazırlanmışdır.

---

## Mündəricat

1. [Giriş və ümumi baxış](#1-giri%C5%9F-v%C9%99-%C3%BCmumi-bax%C4%B1%C5%9F)
2. [Sistem arxitekturası](#2-sistem-arxitekturas%C4%B1)
3. [Dörd iş rejimi](#3-d%C3%B6rd-i%C5%9F-rejimi)
4. [Sorğunun tam yolu (Pipeline) — A-dan Z-yə](#4-sorğunun-tam-yolu-pipeline--a-dan-z-y%C9%99)
5. [Retrieval — Axtarış sisteminin detalları](#5-retrieval--axtar%C4%B1%C5%9F-sisteminin-detallar%C4%B1)
6. [Adaptive Depth — Cavabın dərinliyi](#6-adaptive-depth--cavab%C4%B1n-d%C9%99rinliyi)
7. [Sitatlar və yoxlama sistemi](#7-sitatlar-v%C9%99-yoxlama-sistemi)
8. [Mənbə idarəetməsi](#8-m%C9%99nb%C9%99-idar%C9%99etm%C9%99si)
9. [İstifadəçi interfeysi (UI)](#9-istifad%C9%99%C3%A7i-interfeysi-ui)
10. [Çoxdilli tərcümə boru xətti](#10-%C3%A7oxdilli-t%C9%99rc%C3%BCm%C9%99-boru-x%C9%99tti)
11. [Təhlükəsizlik](#11-t%C9%99hl%C3%BCk%C9%99sizlik)
12. [Yaddaş və sessiyalar](#12-yadda%C5%9F-v%C9%99-sessiyalar)
13. [Konfiqurasiya — .env dəyişənləri](#13-konfiqurasiya--env-d%C9%99yi%C5%9F%C9%99nl%C9%99ri)
14. [RunPod-da quraşdırma və işə salma](#14-runpod-da-qura%C5%9Fd%C4%B1rma-v%C9%99-i%C5%9F%C9%99-salma)
15. [Diaqnostika və monitorinq](#15-diaqnostika-v%C9%99-monitorinq)
16. [Tez-tez verilən suallar (FAQ)](#16-tez-tez-veril%C9%99n-suallar-faq)
17. [Lüğət](#17-l%C3%BCğ%C9%99t)

---

## 1. Giriş və ümumi baxış

### 1.1. Bu nədir?

**AI Köməkçi V4** — sənədlərə əsaslanan, çoxdilli, retrieval-augmented (RAG) süni intellekt köməkçisidir. İstifadəçi kitablar, sənədlər və veb səhifələri yükləyir; sonra suallar verir; sistem bu sənədlərdən cavab tapır, dərin analiz aparır və mənbələri göstərir.

### 1.2. Hansı problemləri həll edir?

| Problem | V4-də həll |
|---|---|
| LLM-lər "halüsinasiya" edir (uydurma cavab verir) | Sadəcə yüklənmiş mənbələrdən cavab verir + hər iddiaya **mənbə** göstərir |
| Sadə sualların cavabı çox uzun və yüklü olur | **Adaptive Depth** — sualın mürəkkəbliyinə görə cavab uzunluğunu avtomatik tənzimləyir |
| LLM yanlış mənbə göstərə bilər | **Citation Verifier** — hər sitatı təkrar yoxlayır, ✓/⚠/✗ qiyməti verir |
| 200+ kitab miqyasında yavaş axtarış | **Hibrid axtarış + Reranker** — diskdə davam edən vektor indeks (ChromaDB) + BM25 + cross-encoder ilə yenidən sıralama |
| Qısa suallar pis cavablar gətirir | **HyDE** — sualdan əvvəl hipotetik cavab yaradıb onu axtarış üçün istifadə edir |
| Bulud LLM-lərə həssas məlumat göndərmək | **Məxfi Rejim** — yerli psevdonimləşdirmə + AES-256 şifrələmə |
| İnternetdən aktual məlumat lazım | **Web Araşdırması Rejimi** — Gemini googleSearch və ya Grok Live Search |

### 1.3. Hansı istifadə halları üçündür?

- Tədqiqat və elmi işlər (kitabxana analizi)
- Hüquqi sənəd analizi
- Hərbi/strateji araşdırma (mövcud korpus 18,000+ chunk hərbi sənədlərdən ibarətdir)
- Korporativ bilik bazası
- Tələbə yardımcısı (universitet materialları)
- Jurnalistika (mənbələrə əsaslanan yazılar)

### 1.4. Hansı dillərdə işləyir?

- **İnterfeys**: Azərbaycan, İngilis, Rus — istifadəçi seçə bilir
- **Daxili emal**: hər zaman İngilis dilində (LLM-lər və embedding modelləri ingiliscə ən yaxşı işləyir)
- **Cavab**: istifadəçinin seçdiyi dildə qaytarılır

---

## 2. Sistem arxitekturası

### 2.1. Ümumi diaqram

```
┌──────────────────────────────────────────────────────────────────┐
│                    BRAUZER (frontend.html)                       │
│       Söhbət · Mənbələr · Tənzimləmələr · 3 dil seçimi           │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ HTTPS + X-API-KEY
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FastAPI (main.py) — Backend                     │
│   /api/chat · /api/sources · /api/health · /api/session/reset    │
└──────┬──────────┬─────────┬─────────┬───────────┬────────────────┘
       │          │         │         │           │
       ▼          ▼         ▼         ▼           ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│services/ │ │services│ │services│ │services│ │ services/│
│ rag.py   │ │reranker│ │ llm.py │ │verifier│ │ crypto.py│
│ (hibrid) │ │(rerank)│ │ (LLM)  │ │ (yox.) │ │ (Fernet) │
└─────┬────┘ └───┬────┘ └────┬───┘ └────┬───┘ └──────────┘
      │          │            │          │
      ▼          ▼            ▼          ▼
  ┌────────┐ ┌────────┐  ┌─────────────────┐
  │Chroma  │ │bge-rer.│  │ Ollama          │
  │DB +    │ │v2-m3   │  │ qwen2.5:14b     │
  │BM25    │ │(GPU)   │  │ nomic-embed-text│
  │(disk)  │ │        │  └─────────────────┘
  └────────┘ └────────┘  ┌─────────────────┐
                         │ Gemini 2.5 Flash│
                         │ Grok-3/4        │
                         └─────────────────┘
```

### 2.2. Komponentlər və faylların sıralı strukturu

```
/workspace/ai-assistant-v3/
├── config.py                 ← Bütün konfiqurasiya (pydantic settings)
├── main.py                   ← FastAPI app, endpointlər
├── frontend.html             ← Tək faylda UI (HTML+JS+CSS)
├── requirements.txt          ← Python asılılıqları
├── .env                      ← Gizli açarlar (.env.example-dən kopyalanır)
├── sources_metadata.json     ← Yüklənmiş mənbələrin reyestri
├── chroma_db/                ← Vektor verilənlər bazası (diskdə)
├── uploads/                  ← Müvəqqəti fayl saxlanışı
└── services/
    ├── __init__.py
    ├── rag.py                ← Hibrid axtarış mühərriki (vector + BM25)
    ├── reranker.py           ← Stage 2 cross-encoder reranker
    ├── llm.py                ← LLM stream, prompt assembly, planner, HyDE
    ├── verifier.py           ← Sitat yoxlayıcı (Gemini Flash)
    └── crypto.py             ← Fernet + pseudonymisation (Secret Mode)
```

### 2.3. Texnologiya stəki

| Layer | Texnologiya | Niyə? |
|---|---|---|
| Backend framework | **FastAPI** (Python 3.11) | Tam asinxron, sürətli, OpenAPI dəstəyi |
| ASGI server | **Uvicorn** | FastAPI üçün standart, yüksək performans |
| Vektor DB | **ChromaDB (PersistentClient)** | Diskdə davam edir, HNSW indeksi, sadə API |
| Keyword search | **rank-bm25** | Sadə, lokal BM25Okapi reallaşdırması |
| Yerli embeddings | **Ollama nomic-embed-text** | 768-ölçülü, 100+ dil dəstəyi, GPU-da sürətli |
| Yerli LLM | **Ollama qwen2.5:14b** | Yaxşı analitik bacarıqlar, ~10GB VRAM |
| Bulud LLM | **Gemini 2.5 Flash, Grok-3/4** | Geniş kontekst, hazır web grounding |
| Reranker | **BAAI/bge-reranker-v2-m3** | Çoxdilli, Apache-2.0, ~250ms top-50 |
| Tərcümə | **Gemini 2.5 Flash** | AZ/EN/RU yüksək keyfiyyət |
| Şifrələmə | **cryptography.fernet** (AES-128 CBC + HMAC) | Standart, audit edilmiş |
| Sənəd parser | **pypdf, beautifulsoup4** | PDF və HTML-dən mətn çıxarma |
| Frontend | **Vanilla JS + Tailwind CDN** | Single-file deployment, bağımlılıqsız |

---

## 3. Dörd iş rejimi

İstifadəçi **⚙ Tənzimləmələr** səhifəsində bu rejimlərdən birini seçir:

### 3.1. 🔒 Oflayn Rejim

**Necə işləyir:** Bütün emal lokal RunPod GPU-da. Ollama (qwen2.5:14b) cavab yazır.

**Üstünlüklər:**
- Heç bir məlumat buludla bölüşülmür
- İnternet tələb etmir (yalnız tərcümə üçün Gemini lazımdır)
- Pulsuz (Cloud LLM ödəyimi yoxdur)

**Çatışmazlıqlar:**
- Cavablar bulud LLM-lərə nisbətən bir az az incə
- Yavaş — qwen2.5:14b ~30-60 saniyə uzun cavab üçün

**Nə vaxt seçməli:** Sürət vacib deyil, məlumat gizliliyi tələb olunur.

### 3.2. 🌐 Onlayn Rejim

**Necə işləyir:** Hibrid axtarış lokal, cavab yazma — Gemini 2.5 Flash və ya Grok (sizin seçiminiz).

**Üstünlüklər:**
- Çox yüksək keyfiyyət (Gemini thinking budget aktiv, geniş bilik)
- Sürətli — 5-15 saniyə tam cavab
- Çoxdilli daha yaxşı

**Çatışmazlıqlar:**
- RAG kontekstiniz buludla bölüşülür
- Hər sual üçün ~$0.001-0.005

**Nə vaxt seçməli:** Standart istifadə — sənədləriniz həssas deyil.

### 3.3. 🤫 Məxfi Rejim

**Necə işləyir:** Buluda göndərməzdən əvvəl mətn lokalca **psevdonimləşdirilir** (e-poçtlar, telefon nömrələri, adlar, IBAN, IP-lər və istifadəçi tərəfindən təyin edilmiş açar sözlər `[[TOKEN_xxx]]` ilə əvəz olunur) və Fernet (AES-256) ilə **şifrələnir**. Cavab gəldikdən sonra lokalca açılır və orijinal mətnlər yenidən yerləşdirilir.

**Üstünlüklər:**
- Buluda gedən mətndə şəxsi məlumat yoxdur
- Şifrələmə açarı yalnız RAM-da yaşayır, heç vaxt diskə yazılmır
- Hər yenidən başlatma yeni açar yaradır

**Çatışmazlıqlar:**
- Maskalanmış mətn bəzən qeyri-təbii görünür və LLM ona reaksiya verə bilər
- Hər zaman bulud lazımdır

**Nə vaxt seçməli:** Tibbi sənədlər, müştəri məlumatları, hüquqi kazuslar, daxili kod.

### 3.4. 🔎 Web Araşdırması Rejimi

**Necə işləyir:** Bulud LLM (Gemini və ya Grok) lokal RAG kontekstini görür **və** öz daxili veb axtarış alətindən istifadə edir. Bu yeni biliklər və aktual hadisələrlə yerli sənədləri birləşdirir.

- **Gemini**: `tools: [{googleSearch: {}}]` paramı aktivdir
- **Grok**: `search_parameters: {mode: "on", sources: [web, news, x]}`

**Üstünlüklər:**
- Yerli korpus + canlı internet birlikdə
- LLM özü qərar verir nə vaxt veb axtarış lazımdır
- Veb mənbələri göstərilir (klik edilə bilən URL-lər)

**Çatışmazlıqlar:**
- Gemini grounding bəzi billing tier-lərdə açıq deyil
- Grok Live Search ayrıca ödəyimlidir (~$25 / 1k axtarış)

**Nə vaxt seçməli:** "Bu gün hansı xəbərlər var?", "X şirkəti ən son nə elan etdi?" kimi suallar.

---

## 4. Sorğunun tam yolu (Pipeline) — A-dan Z-yə

Bir sual yazıldıqdan keçən hər addım:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 0. İstifadəçi Azərbaycanca sualı yazır + dərinlik pilini seçir          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ POST /api/chat (NDJSON stream)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. AZ → EN tərcümə (Gemini Flash)                                       │
│    Niyə? Embedding modelləri, LLM-lər və RAG ingilisdə ən yaxşı işləyir │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. HyDE (yalnız qısa suallar üçün, <30 söz)                             │
│    LLM yalançı 100-200 sözlük "etibarlı görünən" cavab yazır            │
│    Bu mətn vektor axtarışı üçün seed kimi istifadə olunur               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. STAGE 1: Hibrid axtarış (services/rag.py)                            │
│    ┌─ Vektor axtarış (ChromaDB + nomic-embed-text) → 50 namizəd ─┐      │
│    └─ BM25 açar söz axtarışı (rank_bm25)            → 50 namizəd ─┘      │
│       Birləşmə + skor = 0.65 × vektor + 0.35 × BM25                     │
│       Nəticə: ən yaxşı 50 chunk                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. STAGE 2: Reranker (services/reranker.py)                             │
│    bge-reranker-v2-m3 cross-encoder GPU-da işləyir                      │
│    Hər (sual, chunk) cütünü tam diqqətlə yenidən qiymətləndirir         │
│    Nəticə: ən yaxşı 15 chunk                                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. PLAN PASS (services/llm.py — _plan_complete)                         │
│    LLM-ə sual + kontekst verilir, JSON ilə cavab verir:                 │
│    {                                                                    │
│      "question_type": "definition" | "factual" | "comparison" | ...     │
│      "complexity":    "simple" | "moderate" | "deep"                    │
│      "themes": [...]   (yalnız deep üçün 5-7 alt mövzu)                 │
│    }                                                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. DEPTH RESOLUTION                                                     │
│    İstifadəçinin pilini yoxlayır:                                       │
│      - 🤖 Auto → planner-in seçimi qalır                                │
│      - 🪶/📖/🔬 → istifadəçi seçimi planner-i ləğv edir                  │
│    Müvafiq sistem promptu və budget seçilir:                            │
│      simple   → CONCISE_SYSTEM_PROMPT  (150-400 söz, top-5,  1k token)  │
│      moderate → MODERATE_SYSTEM_PROMPT (500-900 söz, top-10, 3k token)  │
│      deep     → DEEP_SYSTEM_PROMPT     (1500+ söz, top-15, 16k token)   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. run_meta hadisəsi UI-a göndərilir                                    │
│    {type: "run_meta", mode, provider, model, depth_chosen, ...}         │
│    UI bunu mavi badge kimi göstərir                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. EXPAND PASS — Cavab yazılır (streamed)                               │
│    Müvafiq sistem promptu + outline + 15 chunk LLM-ə göndərilir         │
│    Token-token cavab gəlir, UI canlı göstərir (▋ kursoru)               │
│    Hər iddiada [Source #N] sitatları olur                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 9. EN → AZ tərcümə (Gemini Flash)                                       │
│    Final cavab istifadəçinin seçdiyi dilə qaytarılır                    │
│    {type: "answer_az", text: "..."}                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 10. CITATION VERIFIER — asinxron yoxlama                                │
│     services/verifier.py:                                               │
│       a) cavabı cümlələrə bölür                                         │
│       b) [Source #N] olan cümlələri "iddia" kimi seçir                  │
│       c) hər iddianı Gemini Flash-a göndərir:                           │
│          "Bu iddianı cited source dəstəkləyirmi?"                       │
│       d) Verdict: supported/partial/unsupported + confidence + qeyd     │
│     {type: "verification", total, supported, partial, unsupported, ...} │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 11. {type: "done"} → axın bağlanır                                      │
│     İstifadəçi UI-da görür:                                             │
│       - Mavi meta badge                                                 │
│       - Final cavab (Markdown ilə)                                      │
│       - 🌍 Veb mənbələr paneli (yalnız web rejimi)                      │
│       - 🔍 N/M iddia yoxlanıldı paneli                                  │
│       - ▶ N mənbə (klik edilə bilən accordion)                          │
│       - 💾 Bilik bazasına saxla düyməsi                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1. Pipeline-ın hər addımının zaman təxmini

| Addım | Tipik vaxt |
|---|---|
| AZ → EN tərcümə | 0.5-1 saniyə |
| HyDE (qısa suallar üçün) | 1-2 saniyə |
| Hibrid axtarış (Stage 1) | 80-200 ms |
| Reranker (Stage 2) | 250 ms |
| Plan pass | 1-3 saniyə |
| Expand pass (Online) | 5-15 saniyə (streamed) |
| Expand pass (Offline) | 20-60 saniyə (streamed) |
| EN → AZ tərcümə | 1-3 saniyə |
| Citation verifier | 5-15 saniyə (8-15 iddia üçün) |
| **CƏMİ (Online, dərin sual)** | **~25-40 saniyə** |
| **CƏMİ (Offline, dərin sual)** | **~40-80 saniyə** |

İstifadəçi ilk tokeni 3-5 saniyəyə görür (streamed), qalan vaxt arxa fonda axır.

---

## 5. Retrieval — Axtarış sisteminin detalları

### 5.1. Vektor (semantik) axtarış necə işləyir?

**Qısaca**: hər mətn parçası 768-ölçülü ədəd vektoruna çevrilir. Yaxın mənalı mətnlər yaxın koordinatlarda yerləşir.

**Addım-addım:**

1. **İndekslənmə vaxtı** (sənəd yüklənəndə):
   - PDF → mətn → 600 simvollu parçalara (chunk) bölünür (80 simvol örtüşmə)
   - Hər chunk Ollama-nın `nomic-embed-text` modelinə göndərilir
   - Geri 768 ədəddən ibarət vektor gəlir
   - ChromaDB bu vektoru saxlayır (HNSW indeksi qurur)

2. **Sorğu vaxtı**:
   - Sual eyni model ilə embed edilir → 768-ölçülü vektor
   - ChromaDB hər saxlanmış vektorla kosinus məsafəsini hesablayır
   - Ən yaxınları qaytarır

**Cosinus oxşarlığı formulu:**
```
similarity = (A · B) / (||A|| × ||B||)
```
Bizim vektorlar L2-normalizə edildiyindən (`||A||=||B||=1`), bu sadəcə nöqtə (dot product) hasilinə bərabərdir.

| `cos θ` | Mana |
|---|---|
| 1.0 | Eyni məna |
| 0.7 | Güclü əlaqəli |
| 0.5 | Zəif əlaqəli |
| 0.0 | Heç bir əlaqə yoxdur |

ChromaDB **cosinus məsafəsi** saxlayır (`1 − cos θ`), kod onu oxşarlığa çevirir:
```python
similarity = max(0.0, 1.0 - cosine_distance)
```

**Üstünlüklər:** Parafrazları, sinonimləri, başqa dildə oxşarları tutur. "Necə inflyasiya pul yığımına təsir edir?" → "qiymət səviyyəsinin artımı depozitlərin real dəyərini azaldır" — sözlər fərqli, məna oxşar.

**Çatışmazlıqlar:** Dəqiq sözləri (xüsusi adlar, model nömrələri) bəzən qaçırır.

### 5.2. BM25 (açar söz) axtarışı necə işləyir?

**BM25** (Best Matching 25) — TF-IDF-nin təkmilləşdirilmiş versiyası. Hansı sözlərin nə qədər nadir olduğunu və hansı sənəddə nə qədər təkrarlandığını ölçür.

**İki əsas komponent:**

#### Term Frequency (TF)
"Sözün bu chunk-da neçə dəfə görünməsi?"
```
"Pişiklər gözəl heyvandır. Mənim pişiyim balıq sevir."
TF("pişik") = 2
TF("balıq") = 1
TF("ev") = 0
```

#### Inverse Document Frequency (IDF)
"Bu söz bütün korpusda nə qədər nadirdir?"
```
IDF(t) = ln((N − n_t + 0.5) / (n_t + 0.5) + 1)
```
- N = ümumi chunk sayı
- n_t = bu sözü əhatə edən chunk sayı

| Söz | n_t | IDF | İzah |
|---|---|---|---|
| "və" | 99,500 | 0.005 | Faydasız (hər yerdə) |
| "inflyasiya" | 2,400 | 3.7 | Məlumatlı |
| "ressentiment" | 3 | 10.4 | Qızıl — demək olar ki, unikal identifikator |

#### BM25 formulu (tam)

$$
\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t,D) \cdot (k_1 + 1)}{f(t,D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$

- $k_1 = 1.5$ — TF saturasiya əmsalı (sözün təkrarlanması get-gedə az əhəmiyyət daşıyır)
- $b = 0.75$ — uzun chunklara cəza əmsalı

### 5.3. Hibrid skor — necə birləşir?

Hər iki axtarıcı ayrı-ayrılıqda **30 namizəd** qaytarır. BM25 skorları per-query normalizə edilir (max-a bölünür) ki, 0..1 aralığında olsun.

**Final skor:**
```
combined = 0.65 × vector_similarity + 0.35 × bm25_normalized
```

`config.py`-də:
```python
HYBRID_VECTOR_WEIGHT = 0.65
HYBRID_BM25_WEIGHT   = 0.35
```

Bu nisbət dəyişdirilə bilər. 65/35 təbii dil sənədləri üçün yaxşı standart hesab olunur.

### 5.4. Reranker — Stage 2

**Problem:** Hibrid skor cəld, lakin yalnız "qısa metrik" verir. İki ayrı encoder (sual üçün və chunk üçün) işlədir — model heç vaxt onları birlikdə görmür.

**Həll:** **Cross-encoder reranker** — `BAAI/bge-reranker-v2-m3`. Sual və chunk **bir input kimi birlikdə** modeldə işlədilir, tam diqqət (full attention) hər iki tərəfi görür.

| Bi-encoder (today, indekslər üçün) | Cross-encoder (Stage 2, reranking üçün) |
|---|---|
| Sual və chunk ayrı-ayrı encode olunur | Birlikdə encode olunur |
| Sürətli (O(1) hər chunk) | Yavaş (~5ms hər cüt) |
| Orta dəqiqlik | Yüksək dəqiqlik |

**Boru xətti:**
```
50 namizəd (Stage 1)
   ↓
[(sual, chunk_1), (sual, chunk_2), ..., (sual, chunk_50)]
   ↓ Cross-encoder GPU-da batch (32 cüt birdən)
50 yeni skor
   ↓ Sıralanır → top-15
   ↓ LLM-ə göndərilir
```

**Niyə vacibdir:** Empirik olaraq +5-15% retrieval dəqiqliyi qaldırır. Ən yaxşı chunk hibrid skor üzrə tez-tez #3 və ya #5 olur — reranker onu #1-ə qaldırır.

**Skor birləşdirmə strategiyası:** `RERANKER_SCORE_MODE = "two_stage"` (default) — hibrid skor yalnız hansı 50-nin reranker-ə getdiyini seçir, reranker skoru isə son sıralamanı verir.

### 5.5. HyDE — Hypothetical Document Embeddings

**Problem:** Qısa suallar pis embed olunur.

"HUMINT nədir?" — 3 söz. Bu mətnin embedding-i "HUMINT haqqında 200 sözlük dolğun mətn"-in embedding-indən fərqlənir.

**Həll:** Sualdan əvvəl LLM-ə deyirik: *"Bu suala yalançı, etibarlı görünən 100-200 sözlük cavab yaz, doğru olub-olmaması fərq etməz, sadəcə doğru terminologiya istifadə et."* Sonra bu **yalançı cavabı** embed edirik, sualı yox.

**BM25 yenə də xam sualı istifadə edir** — orada dəqiq sözlər vacibdir.

**Nə vaxt aktivləşir:**
- `ENABLE_HYDE = True` olduqda
- Sual `HYDE_MIN_QUERY_TOKENS` (30) sözdən qısa olduqda

UI-da `· 💡 HyDE` çipi və "💡 HyDE seed" açılan paneli görünür.

### 5.6. Chunking (Bölmə) strategiyası

**Recursive Character Splitter:**
```python
separators = ["\n\n", "\n", " ", ""]
chunk_size = 600       # simvol
chunk_overlap = 80     # simvol
```

Alqoritm:
1. Mətni `\n\n` (paragraflar) ilə böl
2. Hər parça 600-dən böyükdürsə, `\n` ilə yenidən böl
3. Hələ də böyükdürsə, boşluqlarla böl
4. Son çarə — simvol-simvol kəs

**Overlap (80 simvol)** — qonşu chunklar bir-biri ilə kiçik kəsişməyə malik olur ki, cümlələr ortadan kəsilməsin.

---

## 6. Adaptive Depth — Cavabın dərinliyi

### 6.1. Niyə adaptiv?

V3-də (köhnə) bütün cavablar eyni stil və uzunluq idi. "Pişik nədir?" və "Pişiklərin tarixi və mədəni rolu nədir?" — eyni format. Bu çox vaxt yararsız idi.

V4-də sistem sualın mürəkkəbliyinə görə üç müxtəlif format seçir.

### 6.2. Üç sistem promptu

#### CONCISE_SYSTEM_PROMPT (sadə suallar üçün)

```
Sual sadədir (tərif, fakt axtarışı). Birbaşa cavab ver, struktura bölmə.
- 150-400 söz
- H2 başlıqlarsız ("Executive Summary", "Knowledge Gaps" YOX)
- Hər iddiada [Source #N]
- Kontekst kifayət deyilsə, 1 cümlə ilə de
```

#### MODERATE_SYSTEM_PROMPT (orta suallar üçün)

```
Strukturlu, lakin yığcam.
- 500-900 söz
- Tələb olunan başlıqlar:
  ## Summary
  ## Key Findings
  ## Analysis
- "Caveats" yalnız həqiqi qeydlər varsa
```

#### DEEP_SYSTEM_PROMPT (dərin suallar üçün)

```
Tam analitik tədqiqat.
- 1500-4000 söz
- Tələb olunan başlıqlar:
  ## Executive Summary    (4-8 cümlə)
  ## Key Findings         (5-10 bullet)
  ## Detailed Analysis    (5-7 H3 alt mövzu)
  ## Counter-arguments & Caveats
  ## Knowledge Gaps
  ## Conclusion
- Çoxlu sitat, müqayisə, müxalifət göstər
```

### 6.3. Planner — kim qərar verir?

Cavabdan əvvəl **plan pass** işləyir (eyni LLM-də). O bu JSON-u qaytarır:

```json
{
  "question_type": "definition",
  "complexity":    "simple",
  "themes": []
}
```

| `complexity` | Tipik suallar |
|---|---|
| `simple` | "X nədir", "Y nə vaxt baş verdi" — 0 mövzu |
| `moderate` | "X-i izah et", "qısa müqayisə" — 3-4 mövzu |
| `deep` | "Çoxölçülü müqayisə", "niyə X Y-ə təsir edir" — 5-7 mövzu |

### 6.4. Dərinlik pill bar — istifadəçi nəzarəti

Söhbət ekranının altında 4 düymə:

```
[🤖 Auto] [🪶 Qısa] [📖 Orta] [🔬 Dərin]
```

- **🤖 Auto** (default) — planner-in seçimi qalır
- Digərləri — istifadəçinin seçimi planner-i ləğv edir

Bu seçim localStorage-da saxlanır.

### 6.5. run_meta badge-də necə görünür?

Cavabın yuxarısında mavi çubuqda:
```
🤖 auto → 📖 orta · analysis     ← Planner seçdi (auto)
👤 əl ilə → 🔬 dərin · factual    ← İstifadəçi əl ilə dərin seçdi
```

---

## 7. Sitatlar və yoxlama sistemi

### 7.1. Daxili sitat formatı

Hər iddiadan sonra `[Source #N]` (1-əsaslı). N — `▶ N mənbə` panelindəki sıra nömrəsi.

Misal: 
> "HUMINT insan mənbələrinə əsaslanır [Source #4]. SIGINT siqnal tutmanı əhatə edir [Source #7]."

LLM bunu sistem promptu ilə öyrədilir — hər iddianın mənbəyə bağlanmasını tələb edir.

### 7.2. Klik edilə bilən sitatlar

`linkifyCitations()` funksiyası rendered Markdown-da `[Source #N]` (və `[Sources #2, #7]`) formatlarını tutur və onları `<a class="citation-link">` element-lərinə çevirir.

**Hover** → tooltip ilə:
- Mənbə başlığı
- Skor %
- 320 simvolluq snippet

**Klik** → mənbələr akkordeon-u açılır, müvafiq sıra mavi rəng ilə qısa anim alır.

### 7.3. Citation Verifier — yoxlayıcı

**Niyə lazımdır?** LLM bəzən mənbənin demədiyini iddia edir. Klassik RAG halüsinasiyası.

**Necə işləyir?**

1. Cavab axın bitdikdən sonra **asinxron başlayır** — istifadəçi cavabı oxuyur, yoxlama arxada işləyir
2. Cavabı cümlələrə bölür
3. Yalnız `[Source #N]` olan cümlələri saxlayır (=iddialar)
4. Hər iddianı **Gemini 2.5 Flash**-a göndərir (yalnız Gemini, ucuz və müstəqil):
   ```
   CLAIM: "HUMINT insan mənbələrinə əsaslanır" [Source #4]
   
   CITED SOURCES:
   [Source #4 | Field Manual 2-22.3]
   <chunk mətni>
   
   Cavab JSON:
   {"verdict": "supported|partial|unsupported", "confidence": 0.0-1.0, "note": "..."}
   ```
5. 4 paralel sorğu (`asyncio.Semaphore(4)`)
6. Nəticələri toplayır və UI-a `{type: "verification", ...}` hadisəsi göndərir

### 7.4. Verdict-lərin mənası

| Simvol | Verdict | Mənası |
|---|---|---|
| ✓ | `supported` | Mənbə iddianı tam dəstəkləyir |
| ⚠ | `partial` | Mövzu var, amma iddia bir az kənara çıxır |
| ✗ | `unsupported` | Mənbə bu iddianı saxlamır |

**confidence < 0.6** olduqda `supported` avtomatik `partial`-a düşür.

### 7.5. UI-da necə görünür?

Cavabın altında, mənbələr panelindən yuxarı:
```
▶ 🔍 8/10 supported · ⚠ 2 partial

   ✓ "HUMINT insan mənbələrinə əsaslanır" [#4]
        supported · 92% · birbaşa qeyd edilir
   ⚠ "MI bölmələri mülki şəxslərə qarşı məhdudiyyətlərə malikdir" [#3]
        partial · 65% · mənbə hüquqi uyğunluqdan bəhs edir, mülki şəxslərdən deyil
```

---

## 8. Mənbə idarəetməsi

### 8.1. Üç mənbə tipi

| Tip | Format | Necə əlavə olunur? |
|---|---|---|
| **PDF** | `.pdf` | Drag-drop və ya fayl seçimi |
| **URL** | https://... | URL daxil et + başlıq |
| **Text** | Düz mətn | Başlıq + mətn yapışdır |

### 8.2. Yükləmə prosesi

```
İstifadəçi PDF yükləyir
        ↓
main.py: /api/sources/upload
        ↓
pypdf.extract_text() — səhifə-səhifə mətn çıxarır
        ↓
services/rag.py: HybridSearchEngine.add_source()
        ↓
RecursiveCharacterSplitter — 600/80 chunks
        ↓
Hər chunk Ollama-ya göndərilir → 768-d vektor
        ↓
ChromaDB.add() — vektorlar + metadata diskdə saxlanılır
        ↓
BM25 indeksi yenilənir (yeni chunklar tokenizə olunub əlavə edilir)
        ↓
sources_metadata.json yenilənir
```

### 8.3. Disk persistensi

**ChromaDB** — `/workspace/ai-assistant-v3/chroma_db/` qovluğunda. Bütün vektorlar diskdə qalır, restart-da yenidən embed olunmasına ehtiyac yoxdur.

**BM25** — yaddaşda, lakin app boot-da `_rebuild_bm25_from_disk()` ChromaDB-dən bütün chunkları oxuyur və BM25 indeksini yenidən qurur. Restart sürətli (18,000 chunk üçün ~13 saniyə).

### 8.4. URL emalı

`httpx.AsyncClient` + `beautifulsoup4`:
1. URL fetch olunur (30s timeout, redirects izlənir)
2. PDF-dirsə → `_extract_pdf_text`
3. HTML-dirsə → BeautifulSoup → script/style/nav/footer silinir → təmiz mətn

### 8.5. Silmə

`DELETE /api/sources/{id}`:
1. `sources_metadata.json`-dan reyestr silinir
2. ChromaDB-də `where={"source_id": id}` ilə hansı chunk-ların aid olduğu tapılır
3. Bütün bu chunklar silinir (`collection.delete(ids=...)`)
4. BM25 indeksi yenidən qurulur (bu chunklar atılır)

---

## 9. İstifadəçi interfeysi (UI)

### 9.1. Üç səhifə (Top Navigation)

```
[🧠 AI Köməkçi V4]
[💬 Söhbət] [📁 Mənbələr] [⚙ Tənzimləmələr]
                                     [Dil ⏷] [API açarı]
```

### 9.2. 💬 Söhbət ekranı

- **Başlıq**: Cavab dərinliyi, sessiya ID, mode indikator
- **Söhbət axını**: müraciətlər (mavi baloncuqlar) və cavablar (boz baloncuqlar)
- **Cavabın komponentləri:**
  1. Meta badge (mavi çubuq, hər şeyi göstərir)
  2. İngilis sorğusu (kiçik, görünə bilər)
  3. Final cavab (Markdown, klik edilə bilən sitatlarla)
  4. 🌍 Veb mənbələr (yalnız web rejimi)
  5. 🔍 Yoxlama paneli
  6. ▶ N mənbə (lokal RAG mənbələri)
  7. 💾 Bilik bazasına saxla düyməsi
- **Dərinlik pill bar**: kompozer üstündə
- **Kompozer**: textarea + Göndər düyməsi
- **Top-right düymələr**: ⏹ Dayandır, 🗑 Təmizlə

### 9.3. 📁 Mənbələr ekranı

- **Üst**: səhifə başlığı və "↻ Yenilə" düyməsi
- **Üç kart yan-yana**:
  1. 📄 PDF/Text yüklə (dropzone)
  2. 🌐 URL əlavə et
  3. ✍ Birbaşa mətn əlavə et
- **Alt**: əlavə edilmiş mənbələr siyahısı (silmə düyməsi ilə)
- **Statistika**: N mənbə · M chunk

### 9.4. ⚙ Tənzimləmələr ekranı

1. **İş Rejimi** kartları (4): Oflayn / Onlayn / Məxfi / Web
2. **Bulud Provayderi** radio: Gemini / Grok
3. **Məxfi açar sözlər** input
4. **Sessiya yaddaşı**: ID və sıfırlama düyməsi
5. **Canlı Vəziyyət** (4 status nöqtəsi):
   - 💾 ChromaDB
   - 🦙 Ollama
   - ✨ Gemini
   - 🤖 Grok
6. **Statistika**: Mənbə sayı, chunk sayı, paths
7. **↻ Vəziyyəti yenilə** düyməsi

### 9.5. Üçdilli dəstək

Localizasiya `I18N = { az: {...}, en: {...}, ru: {...} }` lüğəti ilə:
- `data-i18n="key"` — innerHTML üçün
- `data-i18n-ph="key"` — placeholder üçün

Dil dəyişdiriləndə bütün DOM avtomatik yenilənir. Seçim localStorage-da qalır.

### 9.6. Meta badge — hər cavabın "şəxsiyyət vəsiqəsi"

```
🔒 offline · ollama · qwen2.5:14b · top-15 · ✨ reranked · 💡 HyDE · 👤 əl ilə → 🔬 dərin · factual
📋 7 bölmə (açılan)
💡 HyDE seed (açılan)
```

| Çip | Mənası |
|---|---|
| 🔒/🌐/🤫/🔎 | Rejim |
| `provider` | Hansı bulud (Ollama/Gemini/Grok) |
| `model` | Tam model adı |
| `top-N` | LLM-ə neçə chunk getdi |
| `🔬 2-pass` | Plan→Expand pipeline işlədi (deep + temaları var) |
| `✨ reranked` | Cross-encoder Stage 2 işlədi |
| `💡 HyDE` | Hypothetical doc retrieval üçün istifadə edildi |
| `🌍 web` | Veb grounding işlədi (Gemini/Grok) |
| `🤖/👤` | Dərinlik kim seçdi (auto vs manual) |
| `🪶/📖/🔬` | Faktiki dərinlik (simple/moderate/deep) |
| `question_type` | Planner-in sual təsnifatı |

---

## 10. Çoxdilli tərcümə boru xətti

### 10.1. Niyə hər zaman ingiliscə daxili emal?

| Komponent | İngilisdə işləməsinin səbəbi |
|---|---|
| `nomic-embed-text` | İngilis üçün daha geniş təlim datası → daha keyfiyyətli embedding |
| `qwen2.5:14b` | İngilisdə ən yaxşı performans |
| Gemini/Grok | Sistem promptları və terminologiya ingilisdə daha aydın |
| BM25 | Azərbaycan üçün stem-leyici yox, ingilis tokenizasiyası daha standart |
| Reranker (bge-m3) | Çoxdilli amma ingilis korpus üzərində daha çox sınanmış |

### 10.2. Pipeline-da yeri

```
İstifadəçi (AZ)
    ↓
[Gemini AZ→EN tərcümə]
    ↓
RAG + LLM + Verifier (hamısı EN)
    ↓
[Gemini EN→AZ tərcümə]
    ↓
İstifadəçi (AZ)
```

### 10.3. Tərcümə uğursuz olduqda

`translate_az_to_en` və `translate_en_to_az` fallback rejimə malikdir — Gemini xəta versə, sualı/cavabı orijinal dildə qaytarır. Belə olduqda istifadəçi bunu meta badge-də görür.

---

## 11. Təhlükəsizlik

### 11.1. X-API-KEY Gatekeeper

Bütün `/api/*` endpointləri `X-API-KEY` başlığını yoxlayır:

```python
@app.get("/api/health")
async def health(_: str = Depends(require_api_key)):
    ...
```

Açar `.env`-də `API_KEY=` sətirində. Brauzer onu localStorage-da saxlayır və hər sorğuya əlavə edir.

**Səhv açar** → 401 + `"X-API-KEY başlığı tələb olunur və düzgün olmalıdır."`

### 11.2. CORS allowlist

`config.py`-də `ALLOWED_ORIGINS`. **Heç vaxt `*` deyil** — yalnız RunPod proxy URL-i:
```
ALLOWED_ORIGINS=["https://<pod-id>-8000.proxy.runpod.net"]
```

### 11.3. Fernet (AES-256) Secret Mode

`services/crypto.py`:
- **Açar**: `Fernet.generate_key()` boot zamanı, **yalnız RAM-da yaşayır**
- **Diskə yazılmır** (loglara, env-ə, fayllara — heç bir yerə)
- **Restart** = yeni açar
- Hər çağırışda hopefully integrity check (encrypt + decrypt round-trip) edilir

### 11.4. Pseudonymisation

Buluda göndərilən mətndə avtomatik aşkarlanır və `[[TOKEN_xxx]]` yer tutucuları ilə əvəz olunur:
- E-poçt ünvanları
- Telefon nömrələri
- Kredit kartı şəkilli rəqəmlər
- IBAN kodları
- IPv4 ünvanları
- ISO tarixləri
- URL-lər
- Şəxs adları (böyük hərflə başlayan 2 söz)
- İstifadəçi tərəfindən təyin edilmiş açar sözlər

Cavab gəldikdə yer tutucular orijinal mətnlərlə əvəz olunur.

### 11.5. API açar gizliliyi (Gemini)

V4 sonunda kritik dəyişiklik:
- **Köhnə**: `?key=AIzaSyXXX` URL-də → hər xətada açar görünürdü
- **Yeni**: `x-goog-api-key` başlığında → URL təmizdir, xətada açar görünmür

`_sanitise_url()` və `_safe_http_msg()` köməkçiləri əmin edir ki, heç bir error log-da açar görünməsin.

### 11.6. Retry-with-backoff (transient xətalar)

`_retrying_post()` 429, 500, 502, 503, 504 xətalarında və şəbəkə uğursuzluqlarında **eksponensial gecikmə ilə 3 dəfə yenidən cəhd edir**:
- 1-ci uğursuzluq → 0.8s gözlə
- 2-ci uğursuzluq → 1.6s gözlə
- 3-cü uğursuzluq → exception

Bu Gemini-nin tez-tez verdiyi 503-lara qarşı çox yaxşı işləyir.

---

## 12. Yaddaş və sessiyalar

### 12.1. Per-session memory

`services/llm.py: MemoryStore`:
- Hər `session_id` üçün ayrı `ConversationMemory` saxlanır
- Son **10 user/assistant cüt** növbə yaddaşda qalır (sliding window)
- `MEMORY_TURNS = 10` (config-də dəyişdirilə bilər)
- Yalnız RAM-da, diskdə yox

### 12.2. Niyə 10 növbə?

- 5 növbə (V3) — uzun söhbətlərdə kontekst itirdi
- 10 növbə (V4) — yaxşı balans, ~2-4k token əlavə kontekstdə
- Daha çox → token sayı sürətlə artır, LLM diqqəti pozulur

### 12.3. Sıfırlama

İstifadəçi **Tənzimləmələr → 🧹 Yaddaşı sıfırla** düyməsini basır:
- `POST /api/session/{session_id}/reset`
- `MemoryStore`-dan həmin session silinir
- Növbəti sual təmiz başlanır

### 12.4. Sessiya ID

Brauzer yüklənəndə avtomatik yaradılır:
```js
state.sessionId = "s-" + Math.random().toString(36).slice(2, 10)
```
localStorage-da qalır, brauzer bağlandıqda saxlanır.

---

## 13. Konfiqurasiya — .env dəyişənləri

`.env.example`-dən kopyalayın və `.env` adlı edin.

### 13.1. Təhlükəsizlik

| Dəyişən | Default | İzah |
|---|---|---|
| `API_KEY` | (məcburi) | Gatekeeper açarı (`openssl rand -hex 24`) |
| `ALLOWED_ORIGINS` | `["http://localhost:8000"]` | CORS allowlist (JSON array) |

### 13.2. Bulud LLM açarları

| Dəyişən | İzah |
|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| `GROK_API_KEY` | https://console.x.ai |

### 13.3. Yerli model endpointləri

| Dəyişən | Default | İzah |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama HTTP endpoint |
| `OLLAMA_LLM_MODEL` | `qwen2.5:14b` | Yerli LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |

### 13.4. Bulud model identifikatorları

| Dəyişən | Default | İzah |
|---|---|---|
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash` | Söhbət modeli |
| `GEMINI_TRANSLATE_MODEL` | `gemini-2.5-flash` | Tərcümə modeli |
| `GROK_LLM_MODEL` | `grok-2-latest` | Grok modeli (alias dəyişə bilər) |
| `GROK_BASE_URL` | `https://api.x.ai/v1` | Grok API endpoint |

### 13.5. RAG və axtarış

| Dəyişən | Default | İzah |
|---|---|---|
| `CHUNK_SIZE` | 600 | Chunk uzunluğu (simvol) |
| `CHUNK_OVERLAP` | 80 | Chunk-lar arası üst-üstə düşmə |
| `HYBRID_VECTOR_WEIGHT` | 0.65 | Vektor skor çəkisi |
| `HYBRID_BM25_WEIGHT` | 0.35 | BM25 skor çəkisi |
| `RAG_TOP_K` | 40 | Default top-K |
| `RAG_CANDIDATES_PER_INDEX` | 80 | Hər indeksdən gələn namizəd sayı |

### 13.6. Adaptive depth

| Dəyişən | Default | İzah |
|---|---|---|
| `DEFAULT_DEPTH` | `moderate` | Planner uğursuz olduqda fallback |
| `DEPTH_SIMPLE_MAX_TOKENS` | 1024 | ~400 söz |
| `DEPTH_MODERATE_MAX_TOKENS` | 3072 | ~900 söz |
| `DEPTH_DEEP_MAX_TOKENS` | 16384 | ~4000 söz |
| `DEPTH_SIMPLE_TOP_N` | 5 | LLM-ə neçə chunk |
| `DEPTH_MODERATE_TOP_N` | 10 | |
| `DEPTH_DEEP_TOP_N` | 15 | |

### 13.7. Reranker

| Dəyişən | Default | İzah |
|---|---|---|
| `ENABLE_RERANKER` | True | Kill switch |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder |
| `RERANKER_DEVICE` | None (auto) | cuda / cpu |
| `RERANKER_RETRIEVE_K` | 50 | Stage 1 → Stage 2 |
| `RERANKER_TOP_N` | 15 | Stage 2 → LLM (deep üçün) |
| `RERANKER_BATCH_SIZE` | 32 | GPU batch |
| `RERANKER_SCORE_MODE` | `two_stage` | two_stage / replace / blend |
| `RERANKER_WARMUP_ON_STARTUP` | True | Boot-da modeli yüklə |

### 13.8. HyDE

| Dəyişən | Default | İzah |
|---|---|---|
| `ENABLE_HYDE` | True | Aktiv/passiv |
| `HYDE_MIN_QUERY_TOKENS` | 30 | Bundan qısa suallarda aktivləşir |
| `HYDE_MAX_OUTPUT_TOKENS` | 384 | ~200 söz |

### 13.9. Verifier

| Dəyişən | Default | İzah |
|---|---|---|
| `ENABLE_VERIFIER` | True | Aktiv/passiv |
| `VERIFIER_MAX_CLAIMS` | 20 | Maksimum yoxlanılan iddia sayı |
| `VERIFIER_MIN_CONFIDENCE` | 0.6 | Bundan aşağı → "partial" |
| `VERIFIER_CHUNK_PREVIEW_CHARS` | 800 | Judge-ə göstərilən chunk uzunluğu |

### 13.10. Web mode

| Dəyişən | Default | İzah |
|---|---|---|
| `ENABLE_WEB_MODE` | True | Aktiv/passiv |
| `WEB_MAX_RESULTS` | 10 | Grok max search results |

### 13.11. Digər

| Dəyişən | Default | İzah |
|---|---|---|
| `MEMORY_TURNS` | 10 | Yaddaşda saxlanan növbələr |
| `MAX_OUTPUT_TOKENS` | 16384 | Ümumi ceiling |
| `TEMPERATURE` | 0.6 | LLM yaradıcılıq |
| `GEMINI_THINKING_BUDGET` | 4096 | Gemini 2.5 extended reasoning |
| `OLLAMA_NUM_CTX` | 16384 | Ollama context window |

---

## 14. RunPod-da quraşdırma və işə salma

### 14.1. Pod yaratma

1. **runpod.io → Pods → Deploy**
2. **Template**: `RunPod PyTorch 2.x`
3. **GPU**: minimum 16 GB VRAM (RTX A4000, RTX 4090, A5000)
4. **Container Disk**: 20 GB
5. **Volume Disk**: 50+ GB, mount-point `/workspace`
6. **HTTP Ports**: `8000` əlavə et
7. **Deploy**

### 14.2. İlk quraşdırma

```bash
# 1. Ollama quraşdır
apt-get update && apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh

# 2. Modelləri /workspace-ə yönləndir (restart-da itməsin)
mkdir -p /workspace/ollama-models
export OLLAMA_MODELS=/workspace/ollama-models

# 3. Ollama-nı başlat
nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 2

# 4. Modelləri çək
ollama pull qwen2.5:14b           # ~9 GB
ollama pull nomic-embed-text      # ~280 MB

# 5. Kodları clone et
mkdir -p /workspace/ai-assistant-v3
cd /workspace/ai-assistant-v3
git clone https://github.com/RaziMirzazada/AIAssistant.git .

# 6. Python venv (workspace-də qalsın)
python3 -m venv /workspace/venv
echo "source /workspace/venv/bin/activate" >> ~/.bashrc
source ~/.bashrc

# 7. Asılılıqları quraşdır (~5-15 dəqiqə torch üçün)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 8. .env hazırla
cat > .env <<EOF
API_KEY=$(openssl rand -hex 24)
GEMINI_API_KEY=YOUR_GEMINI_KEY
GROK_API_KEY=YOUR_GROK_KEY
ALLOWED_ORIGINS=["https://<POD-ID>-8000.proxy.runpod.net"]
EOF

# 9. Tətbiqi işə sal
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /workspace/app.log 2>&1 &
sleep 30 && tail -n 30 /workspace/app.log
```

### 14.3. Restart skripti (pod yenidən başladıqda)

`/workspace/start.sh`:
```bash
#!/bin/bash
export OLLAMA_MODELS=/workspace/ollama-models

# Ollama re-install (container disk-i restart-da silinir)
if ! command -v ollama >/dev/null; then
    apt-get update -qq && apt-get install -y -qq zstd
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Ollama başlat
pkill -f "ollama serve" 2>/dev/null; sleep 1
nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 2

# Tətbiqi başlat (venv /workspace-də qalır)
source /workspace/venv/bin/activate
cd /workspace/ai-assistant-v3
pkill -f "uvicorn main:app" 2>/dev/null; sleep 1
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /workspace/app.log 2>&1 &
sleep 5
tail -n 15 /workspace/app.log
```

Yenidən başladıqdan sonra sadəcə: `/workspace/start.sh`

### 14.4. Kodu yeniləmək

```bash
cd /workspace/ai-assistant-v3
git pull --no-rebase --no-edit origin main
pkill -f "uvicorn main:app" 2>/dev/null; sleep 1
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /workspace/app.log 2>&1 &
tail -f /workspace/app.log   # Ctrl+C "Uvicorn running" göründükdən sonra
```

---

## 15. Diaqnostika və monitorinq

### 15.1. /api/health endpoint

```bash
KEY=$(grep ^API_KEY= /workspace/ai-assistant-v3/.env | cut -d= -f2-)
curl -s -H "X-API-KEY: $KEY" http://127.0.0.1:8000/api/health | python3 -m json.tool
```

Çıxış:
```json
{
  "status": "ok",
  "engine": {
    "sources": 44,
    "chunks": 18775,
    "chroma_path": "/workspace/ai-assistant-v3/chroma_db"
  },
  "services": {
    "ollama": true,
    "gemini": true,
    "grok":   true
  },
  "workspace": "/workspace/ai-assistant-v3"
}
```

### 15.2. app.log oxuma

```bash
tail -f /workspace/app.log

# Yalnız xətalar
grep -i "error\|exception\|traceback" /workspace/app.log | tail -30

# Hər sorğunun xülasəsi
grep "chat()" /workspace/app.log | tail -10
grep "run_meta" /workspace/app.log | tail -10
grep "verification" /workspace/app.log | tail -10
```

### 15.3. Tipik log axını

```
[INFO] ai-assistant-v3 :: Booting HybridSearchEngine ...
[INFO] services.rag :: BM25 rebuilt from disk: 18775 chunks indexed.
[INFO] ai-assistant-v3 :: Engine ready: {'sources': 44, ...}
[INFO] ai-assistant-v3 :: Warming up reranker 'BAAI/bge-reranker-v2-m3' …
[INFO] services.reranker :: Reranker 'BAAI/bge-reranker-v2-m3' ready on cuda.
[INFO] ai-assistant-v3 :: Reranker warmup complete — Stage 2 will run on every query.
INFO:     Uvicorn running on http://0.0.0.0:8000

# Bir sorğu zamanı:
[INFO] ai-assistant-v3 :: chat() session=s-abc mode=online provider=gemini depth_override=auto msg_chars=42
[INFO] services.llm :: Plan: complexity=moderate qtype=analysis themes=4 (mode=online provider=gemini).
[INFO] ai-assistant-v3 :: run_meta: {'mode': 'online', 'provider': 'gemini', ...}
[INFO] ai-assistant-v3 :: verification: 7 total · 6 supported · 1 partial · 0 unsupported
```

### 15.4. UI-da diaqnostika

Hər cavabın yuxarısındakı **meta badge** ground truth-dur. Əgər istifadəçi düşünür ki, mode dəyişməyib, badge-i göstərmək kifayətdir.

Tənzimləmələrdə **Canlı Vəziyyət** paneli:
- 🟢 yaşıl = işləyir
- 🔴 qırmızı = işləmir (Gemini açarı bitib, Grok billing yoxdur, Ollama dayanıb)
- 🟡 sarı = yoxlanılır

---

## 16. Tez-tez verilən suallar (FAQ)

### "Cavabım Azərbaycanca deyil, ingiliscə qayıdır"
Gemini EN→AZ tərcümə uğursuz olub. Səbəblər:
- Mövzu safety filter-i tetik etdi (hərbi, tibbi)
- Gemini 503 verdi (transient)
- Çox uzun cavab idi (timeout)

Həll: bir az sonra yenidən cəhd. Davam edirsə, mövzunu fərqli ifadə edin.

### "Cavab çox uzundur" / "Cavab çox qısadır"
Dərinlik pilini dəyişin:
- 🪶 Qısa — sadə cavab
- 📖 Orta — strukturlu
- 🔬 Dərin — tam tədqiqat
- 🤖 Auto (default) — planner avtomatik qərar verir

### "HyDE niyə görünmür?"
HyDE yalnız **qısa suallar** (<30 söz) üçün işləyir. Uzun sualınız varsa, vektor axtarış onsuz da yaxşı işləyəcək. `HYDE_MIN_QUERY_TOKENS=50` qoyaraq daha geniş aktivləşdirə bilərsiniz.

### "Verifier paneli '12 partial' deyir"
Gemini 503-ə düşüb və bütün judge çağırışları uğursuz olub (`verifier error — could not check`). V4 b470717 commitindən sonra retry-with-backoff bunu avtomatik həll edir.

### "502 Bad Gateway alıram"
- Uvicorn dayanıb → `pgrep -fa uvicorn` ilə yoxla
- Port 8000 expose olmayıb → RunPod → Edit Pod → HTTP ports
- App boot edir → `tail -f /workspace/app.log` izlə

### "Mənbə əlavə edə bilmirəm"
- API açar səhv → `.env`-də olan ilə UI-da olan eyni olmalıdır
- Disk dolu → `df -h /` yoxla
- ChromaDB lock → uvicorn-u restart et

### "Grok 404 verir"
`grok-2-latest` alias-ı sizin hesabınızda mövcud deyil. Aktiv modelləri yoxlayın:
```bash
curl -H "Authorization: Bearer $GROK_KEY" https://api.x.ai/v1/models
```
Sonra `.env`-də `GROK_LLM_MODEL=` dəyəri uyğun model ilə yeniləyin (məs. `grok-3-mini`).

### "Niyə Offline rejimdə Gemini lazımdır?"
Tərcümə üçün. AZ↔EN köprüsü daxili emal üçündür. İstəyirsinizsə tam offline (heç bir bulud çağırışı) — UI-da İngilis dilini seçin və `translate_az_to_en/translate_en_to_az` çağırışlarını skip etmək üçün kiçik bir kod dəyişikliyi tələb olunur.

### "İstifadə xərci nə qədərdir?"
| İş | Təxmin |
|---|---|
| Offline rejim | $0 (yalnız RunPod GPU) |
| Online + Gemini | ~$0.001-0.005 / sual |
| Online + Grok | ~$0.005-0.02 / sual |
| Web mode + Grok Live Search | +$25 / 1k axtarış |
| Verifier (8-15 iddia/cavab) | ~$0.001-0.003 / cavab |
| RunPod GPU (A4000) | ~$0.20-0.30 / saat |

---

## 17. Lüğət

| Termin | İzah |
|---|---|
| **Chunk** | Sənədin kiçik mətn parçası (600 simvol) |
| **Embedding** | Mətnin 768-ölçülü ədəd vektoru |
| **Vector DB** | Embedding-lərin saxlandığı verilənlər bazası |
| **Cosine similarity** | İki vektorun bucağının kosinusu (oxşarlıq ölçüsü) |
| **BM25** | Açar söz əsaslı axtarış alqoritmi (TF-IDF təkmilləşdirilmişi) |
| **Hybrid search** | Vektor + BM25 nəticələrinin birləşdirilməsi |
| **Cross-encoder** | Sual və chunk-ı birlikdə işləyən rerank modeli |
| **HNSW** | Yaxınlıq axtarışı üçün qraf indeks |
| **HyDE** | Hypothetical Document Embeddings |
| **RAG** | Retrieval-Augmented Generation |
| **LLM** | Large Language Model (böyük dil modeli) |
| **Token** | Sözün bir hissəsi (~3-4 simvol) |
| **Context window** | LLM-in bir dəfədə oxuya bilən maksimum token sayı |
| **Streaming** | Cavabı token-token canlı çatdırmaq |
| **NDJSON** | Newline-delimited JSON (axın protokolu) |
| **Fernet** | AES-128 CBC + HMAC simmetrik şifrələmə standartı |
| **Pseudonymisation** | Şəxsi məlumatları yer tutucularla əvəz etmə |
| **Grounding** | LLM cavabının xarici mənbələrə bağlanması |
| **Verifier** | İddiaları mənbəyə qarşı yoxlayan ikinci LLM |
| **Reranker** | İlkin axtarış nəticələrini yenidən sıralayan model |
| **Outline** | Cavabın strukturlu planı (5-7 alt mövzu) |
| **Plan pass** | Cavabdan əvvəl outline yaradan LLM çağırışı |
| **Expand pass** | Outline-a görə tam cavabı yazan LLM çağırışı |
| **Two-pass** | Plan + Expand birgə |
| **Run meta** | Hər cavab üçün diaqnostik metadata |

---

## Əlavə resurslar

- **Mənbə kodu**: https://github.com/RaziMirzazada/AIAssistant
- **README**: layihənin əsas oxu faylı
- **DESIGN docs**: yol xəritəsi və gələcək addımlar (sənəd növbəti versiyalarda)

---

**Versiya:** V4 (Phase 1 + Phase 1.5)  
**Son yeniləmə:** 2026-05-27  
**Texniki dəstək:** GitHub Issues üzərindən
