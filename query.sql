SELECT
  p.publication_number,
  tl.text AS title_text,
  tl.language AS title_language,
  p.application_number,
  p.country_code,
  p.publication_date,
  p.filing_date,
  p.grant_date,
  p.priority_date,
  (SELECT STRING_AGG(ih.name, ' | ')
     FROM UNNEST(p.inventor_harmonized) AS ih
  )               AS inventor_names,
  (SELECT STRING_AGG(ih.country_code, ' | ')
     FROM UNNEST(p.inventor_harmonized) AS ih
  )               AS inventor_countries,
  (SELECT STRING_AGG(ah.name, ' | ')
     FROM UNNEST(p.assignee_harmonized) AS ah
  )               AS assignee_names,
  (SELECT STRING_AGG(ah.country_code, ' | ')
     FROM UNNEST(p.assignee_harmonized) AS ah
  )               AS assignee_countries,
  (SELECT STRING_AGG(ipc.code, ' | ')
     FROM UNNEST(p.ipc) AS ipc
  )               AS ipc_codes,
  (SELECT STRING_AGG(cpc.code, ' | ')
     FROM UNNEST(p.cpc) AS cpc
  )               AS cpc_codes
FROM
  `patents-public-data.patents.publications` AS p
CROSS JOIN
  UNNEST(p.title_localized) AS tl
WHERE
  tl.language = 'en'
  AND p.publication_date BETWEEN {yyyy}{mm}00 AND {yyyy}{mm}32
  AND p.country_code = 'WO'
ORDER BY
  p.publication_date ASC
