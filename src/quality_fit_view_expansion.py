"""Fixed metadata-only SEM361 role expansion; no label-dependent selection."""
import math


def expand_roles(original, extras, cameras, radius):
    roles=[]
    for role in original:
        if not role['evaluation'] or set(role['fit']) & (set(role['evaluation']) | set(role['guard'])):
            raise ValueError('Invalid original role')
        added=sorted(extras[role['fold']])
        if len(set(added)) != len(added) or set(added) & (set(role['fit']) | set(role['evaluation']) | set(role['guard'])):
            raise ValueError('New names overlap original roles or repeat')
        guard=[n for n in added if min(math.dist(cameras[n].center,cameras[t].center) for t in role['evaluation']) < radius]
        fit=[n for n in added if n not in guard]
        roles.append({**role,'fit':role['fit']+fit,'guard':role['guard']+guard})
    return roles


def coverage(original, expanded, cameras):
    out={}
    for old,new in zip(original,expanded):
        if old['id'] != new['id'] or old['evaluation'] != new['evaluation']:
            raise ValueError('Coverage cannot change evaluation')
        rows=[]
        for n in old['evaluation']:
            before=min(math.dist(cameras[n].center,cameras[t].center) for t in old['fit'])
            after=min(math.dist(cameras[n].center,cameras[t].center) for t in new['fit'])
            if after > before+1e-12:
                raise ValueError('Expansion lost a training view')
            rows.append(dict(name=n,nearest_fit_before=before,nearest_fit_after=after))
        out[old['id']]=dict(rows=rows,closer_evaluation_views=sum(r['nearest_fit_after'] < r['nearest_fit_before']-1e-12 for r in rows),
                            added_fit_views=len(new['fit'])-len(old['fit']))
    return out
