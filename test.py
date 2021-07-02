# import plotly.express as px
# import pandas as pd
#
# df = px.data.iris()
# df = pd.read_csv("/home/quangdq/Kaid/NSFW/nsfw-api/app/test3.csv")
# cls = [str(i) for i in range(5)]
# cls.append('models')
# fig = px.parallel_coordinates(df[cls], color='models',
#                              color_continuous_scale=px.colors.diverging.Tealrose,
#                              color_continuous_midpoint=0.5)
# fig.show()

from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
df = px.data.iris()
df = pd.read_csv("/home/quangdq/Kaid/NSFW/nsfw-api/app/test3.csv")
cls = [str(i) for i in range(5)]
cls.append('models')
df=df[cls]
features = df.loc[:, :'4']

# tsne = TSNE(n_components=3, random_state=0)
# projections = tsne.fit_transform(features)
#
# fig = px.scatter_3d(
#     projections, x=0, y=1,z=2,
#     color=df.models, labels={'color': 'models'}
# )
# fig.show()
from umap import UMAP
import plotly.express as px

#df = px.data.iris()

#features = df.loc[:, :'petal_width']

umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_3d = UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(features)
proj_3d = umap_3d.fit_transform(features)

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.species, labels={'color': 'species'}
)
fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=df.species, labels={'color': 'species'}
)
fig_3d.update_traces(marker_size=5)

fig_2d.show()
fig_3d.show()