-- Crea las vistas necesarias para la app

CREATE VIEW IF NOT EXISTS v_comuna_region AS
    SELECT DISTINCT
        comuna,
        region,
        region_num
    FROM resultados_servel
    WHERE comuna IS NOT NULL
      AND region IS NOT NULL;

CREATE VIEW IF NOT EXISTS v_presidencial_comuna AS
    SELECT
        anio,
        vuelta,
        'TRICEL' AS fuente,
        region_num,
        region,
        comuna_num,
        comuna,
        candidato,
        partido,
        votos,
        votos_nulos,
        votos_blancos,
        total_votos,
        inscritos
    FROM resultados_tricel
    UNION ALL
    SELECT
        anio,
        vuelta,
        'SERVEL' AS fuente,
        region_num,
        region,
        NULL       AS comuna_num,
        comuna,
        candidato,
        partido,
        votos,
        NULL       AS votos_nulos,
        NULL       AS votos_blancos,
        NULL       AS total_votos,
        NULL       AS inscritos
    FROM resultados_servel
    WHERE tipo = 'presidencial'
    UNION ALL
    SELECT
        s.anio,
        s.vuelta,
        'SERVEL_2025' AS fuente,
        cr.region_num,
        cr.region,
        NULL       AS comuna_num,
        s.comuna,
        s.candidato,
        NULL       AS partido,
        s.votos,
        NULL       AS votos_nulos,
        NULL       AS votos_blancos,
        NULL       AS total_votos,
        NULL       AS inscritos
    FROM servel_presid_2025_comuna s
    LEFT JOIN v_comuna_region cr
           ON s.comuna = cr.comuna;

CREATE VIEW IF NOT EXISTS v_presidencial_coalicion AS
    SELECT
        v.anio,
        v.vuelta,
        v.fuente,
        v.region_num,
        v.region,
        v.comuna_num,
        v.comuna,
        v.candidato,
        v.partido,
        cp.coalicion,
        v.votos,
        v.votos_nulos,
        v.votos_blancos,
        v.total_votos,
        v.inscritos
    FROM v_presidencial_comuna v
    LEFT JOIN coalicion_partido cp
      ON v.partido = cp.partido
     AND v.anio BETWEEN cp.anio_inicio AND cp.anio_fin;
