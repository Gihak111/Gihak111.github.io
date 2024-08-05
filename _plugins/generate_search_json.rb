require 'json'

module Jekyll
  class GenerateSearchJson < Generator
    safe true
    priority :low

    def generate(site)
      # 검색에 사용할 포스트 데이터 추출
      search_data = site.posts.docs.map do |post|
        {
          "title" => post.data['title'],
          "url" => post.url
        }
      end

      # search.json 파일 생성 및 저장
      search_file = File.join(site.dest, 'search.json')
      File.open(search_file, 'w') do |file|
        file.write(JSON.pretty_generate(search_data))
      end
    end
  end
end
