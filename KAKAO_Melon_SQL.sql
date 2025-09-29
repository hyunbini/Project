SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP DATABASE my_music_db;
CREATE DATABASE my_music_db DEFAULT CHARACTER SET utf8mb4;
USE my_music_db;

/* =========================
   Album : 앨범 메타
========================= */
CREATE TABLE Album (
  album_id     BIGINT UNSIGNED PRIMARY KEY NOT NULL,
  label        VARCHAR(200) NULL,
  title        VARCHAR(300) NOT NULL,
  release_date DATE NULL,                       -- 발매일
  album_type   VARCHAR(20) NULL,                -- album/ep/single/live/ost
  created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY idx_album_title (title),
  KEY idx_album_type (album_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Artist : 아티스트 메타
========================= */
CREATE TABLE Artist (
  artist_id     BIGINT UNSIGNED PRIMARY KEY NOT NULL,
  name_primary  VARCHAR(200) NOT NULL,          -- 사진의 (15)는 200으로 보정
  country       CHAR(2) NULL,                   -- ISO-3166-1 alpha-2
  debut         DATE NULL,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Track : 곡 메타(버전/원곡 연결)
========================= */
CREATE TABLE track (
  track_id      BIGINT UNSIGNED PRIMARY KEY NOT NULL,
  isrc          VARCHAR(15) NULL,
  title         VARCHAR(300) NOT NULL,
  album_id      BIGINT UNSIGNED NULL,
  release_date          DATE NULL,                      -- 발매일
  duration      INT NULL,                       -- ms 또는 s (문서에 맞춰 표기)
  language_code CHAR(2) NULL,                   -- 언어
  version       VARCHAR(20) NULL,               -- original/remix/live/remaster
  original_id   BIGINT UNSIGNED NULL,           -- 원곡 참조(자기참조)
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_track_isrc (isrc),
  KEY idx_track_title (title),
  KEY idx_track_version (version),
  KEY idx_track_album (album_id),
  CONSTRAINT fk_track_album    FOREIGN KEY (album_id)    REFERENCES album(album_id),
  CONSTRAINT fk_track_original FOREIGN KEY (original_id) REFERENCES track(track_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Credit : Track–Artist 역할 단위 연결(N:N)
========================= */
CREATE TABLE Credit (
  track_id     BIGINT UNSIGNED NOT NULL,
  artist_id    BIGINT UNSIGNED NOT NULL,
  artist_role         VARCHAR(30) NOT NULL,            -- 사진기준 30
  credit_order INT NULL,
  data_source       VARCHAR(50) NULL,
  confidence   DECIMAL(2,1) NOT NULL,               -- 0.0 ~ 1.0
  created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (track_id, artist_id, artist_role),
  KEY idx_credit_role (artist_role),
  CONSTRAINT fk_credit_track  FOREIGN KEY (track_id)  REFERENCES track(track_id),
  CONSTRAINT fk_credit_artist FOREIGN KEY (artist_id) REFERENCES artist(artist_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Genre_tree : 장르 사전(계층)
========================= */
CREATE TABLE Genre_tree (
  genre_id         BIGINT UNSIGNED PRIMARY KEY NOT NULL,
  genre_name             VARCHAR(300) NOT NULL,
  parent_genre_id  BIGINT UNSIGNED NULL,
  UNIQUE KEY uq_genre_name (genre_name),
  CONSTRAINT fk_genre_parent FOREIGN KEY (parent_genre_id) REFERENCES genre_tree(genre_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Genre : 곡–장르 태깅(N:N)
========================= */
CREATE TABLE Genre (
  track_id    BIGINT UNSIGNED NOT NULL,
  genre_id    BIGINT UNSIGNED NOT NULL,
  confidence  DECIMAL(2,1) NULL,                -- 0.0 ~ 1.0
  data_source      VARCHAR(50) NULL,
  PRIMARY KEY (track_id, genre_id),
  CONSTRAINT fk_genre_track  FOREIGN KEY (track_id) REFERENCES track(track_id),
  CONSTRAINT fk_genre_dict   FOREIGN KEY (genre_id) REFERENCES genre_tree(genre_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Relation : 곡–곡 관계(샘플/커버/리믹스/OST…)
========================= */
DROP TABLE IF EXISTS Relation;
CREATE TABLE relation (
  track_id         BIGINT UNSIGNED NOT NULL,
  relation_type    VARCHAR(30) NOT NULL,        -- 사진기준 30
  related_track_id BIGINT UNSIGNED NOT NULL,
  note             VARCHAR(300) NULL,
  source           VARCHAR(50)  NULL,
  PRIMARY KEY (track_id, relation_type, related_track_id),
  KEY idx_rel_related (related_track_id, relation_type),
  CONSTRAINT fk_rel_track          FOREIGN KEY (track_id)         REFERENCES track(track_id),
  CONSTRAINT fk_rel_related_track  FOREIGN KEY (related_track_id) REFERENCES track(track_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Lyrics : 가사 본문/언어/저작권
========================= */
CREATE TABLE Lyrics (
  track_id        BIGINT UNSIGNED PRIMARY KEY NOT NULL,  -- Track:Lyrics = 1:1
  text_raw        MEDIUMTEXT NULL,
  text_language   CHAR(2) NULL,                 -- 사진 표기 그대로
  official_flag   BOOLEAN NULL,
  copyright       VARCHAR(200) NULL,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FULLTEXT KEY ftx_lyrics_text (text_raw),
  CONSTRAINT fk_lyrics_track FOREIGN KEY (track_id) REFERENCES track(track_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Provenance : 출처 기록
========================= */
CREATE TABLE Provenance (
  prov_id      BIGINT UNSIGNED PRIMARY KEY NOT NULL,
  entity_type  VARCHAR(15) NOT NULL,            -- track/artist/album/genre/lyrics…
  entity_id    BIGINT UNSIGNED NOT NULL,
  source_name  VARCHAR(50) NOT NULL,            -- 레이블/API/CMS/모델 등
  fetch_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  license      VARCHAR(30) NULL,                -- 공식/비공식/CC
  KEY idx_prov_entity (entity_type, entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========================
   Quality : 품질/검수 상태
========================= */
CREATE TABLE Quality (
  entity_id        BIGINT UNSIGNED NOT NULL,
  entity_type      VARCHAR(15) NOT NULL,
  data_status           VARCHAR(30) NOT NULL,        -- raw/auto/reviewed/locked 등
  quality_score    TINYINT UNSIGNED NULL,       -- 사진은 BIGINT였으나 0~100 점수에 맞춰 보정
  last_review_time TIMESTAMP NULL,              -- 사진의 DATE를 TIMESTAMP로 보정
  reviewer_id      INT NULL,
  issue            CHAR(2) NULL,                -- 사진 표기 유지(이슈 코드)
  PRIMARY KEY (entity_type, entity_id),
  KEY idx_quality_status (data_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;
