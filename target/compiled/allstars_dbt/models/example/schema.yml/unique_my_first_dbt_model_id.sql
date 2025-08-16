
    
    

select
    id as unique_field,
    count(*) as n_records

from "analytics_dev"."csarat"."analytics_prd__sch_rpt__first_model"
where id is not null
group by id
having count(*) > 1


