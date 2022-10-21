#!/usr/bin/env ruby

# parser for WTHOR file
# Copyrihgt (c) tomykaira 2012
# Lisence: MIT

filenames = ARGV.length == 0 ? ["WTH_2022.wtb"] : ARGV
CLEAN = true

filenames.each do |fn|
  # outputfile = fn.sub(/\.wtb$/i, ".txt")
  outputfile = "all.txt"
  # next if File.exist?(outputfile)
  File.open(fn, "rb") do |f|
    puts fn
    File.open(outputfile, "a+") do |out|
      header = f.read(16)
      hands = header[4...8].unpack("L")[0]

      hands.times do
        game = f.read(68)
        break if game.size < 68
        theo, act = game[6...8].unpack("C*")
        hands = game[8...68].unpack("C*")
        begin
          line = hands.map { |h|  "_ABCDEFGH"[h % 10] + (h / 10).to_s  }.join('').gsub(/(_0)+\Z/, '')
          if !CLEAN || !line.include?("_0")
            out.puts line
          end
        rescue => e
          p "error in file, ignoreing", hands
          break
        end
      end
    end
  end
end
