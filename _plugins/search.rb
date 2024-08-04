module Jekyll
  class SearchIndexGenerator < Generator
    safe true
    priority :low

    def generate(site)
      search_index = site.config['search']
      if search_index
        search_index_data = []

        site.posts.docs.each do |post|
          search_index_data << {
            'title' => post.data['title'],
            'url' => post.url,
            'content' => post.content
          }
        end

        File.open(File.join(site.dest, 'search.json'), 'w') do |f|
          f.write(JSON.pretty_generate(search_index_data))
        end
      end
    end
  end
end
